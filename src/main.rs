use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use glam::Vec3;

use rasterizer::{
    config::Config,
    cpu,
    geometry::{GltfAsset, prepare_scene},
    lighting::{DirectionalLight, Light, RenderLight, collect_lights},
    material::load_materials,
    pipeline::{FragmentStage, Pipeline, Shading},
    shadow::prepare_lights,
    trace::TraceFragment,
};

const USAGE: &str = "Usage: rasterizer <scene-name|path.gltf|path.glb> [--camera-light] [--cuda] [--trace]\n\nWithout arguments, list bundled scenes.\n--camera-light: add a white, unshadowed directional light aligned with the camera (off by default).\n--cuda: keep vertex, rasterization, fragment and shadow work on CUDA (requires the cuda feature).\n--trace: also write the CPU fragment shader's operations to output/<scene>.trace.\nRender: cargo run --release -- living-room";

fn scene_paths() -> io::Result<Vec<PathBuf>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut directories = vec![root.join("assets/scenes")];
    let mut scenes = Vec::new();

    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                directories.push(path);
            } else if file_type.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        ext.eq_ignore_ascii_case("gltf") || ext.eq_ignore_ascii_case("glb")
                    })
            {
                scenes.push(path);
            }
        }
    }

    scenes.sort();
    Ok(scenes)
}

fn scene_name(path: &Path) -> &OsStr {
    path.parent().and_then(Path::file_name).unwrap_or_default()
}

fn list_scenes() -> io::Result<()> {
    let scenes = scene_paths()?;
    if scenes.is_empty() {
        println!("No bundled scenes found.");
    } else {
        println!("Available scenes:");
        for scene in scenes {
            println!("  {}", scene_name(&scene).to_string_lossy());
        }
    }
    Ok(())
}

fn resolve_scene(path: &Path) -> io::Result<PathBuf> {
    if path.is_file() || path.components().count() != 1 || path.extension().is_some() {
        return Ok(path.to_owned());
    }

    let matches: Vec<_> = scene_paths()?
        .into_iter()
        .filter(|scene| scene_name(scene) == path.as_os_str())
        .collect();
    match matches.as_slice() {
        [scene] => Ok(scene.clone()),
        [] => Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Unknown scene '{}'. Run without arguments to list scenes.",
                path.display()
            ),
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Multiple assets match '{}'; use a file path.",
                path.display()
            ),
        )),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut camera_light = false;
    let mut use_cuda = false;
    let mut trace = false;
    let mut positional = Vec::new();
    for arg in env::args_os().skip(1) {
        if arg == "--camera-light" {
            camera_light = true;
        } else if arg == "--cuda" {
            use_cuda = true;
        } else if arg == "--trace" {
            trace = true;
        } else if arg == "--help" || arg == "-h" {
            println!("{USAGE}");
            return Ok(());
        } else if arg.to_string_lossy().starts_with('-') {
            return Err(format!("Unknown option '{}'.\n\n{USAGE}", arg.to_string_lossy()).into());
        } else {
            positional.push(arg);
        }
    }
    let mut args = positional.into_iter();
    let Some(path) = args.next() else {
        if camera_light || use_cuda || trace {
            return Err(format!("Expected a scene.\n\n{USAGE}").into());
        }
        list_scenes().map_err(|error| format!("Failed to list scenes: {error}"))?;
        return Ok(());
    };

    if args.next().is_some() {
        return Err(format!("Expected one scene name or asset path.\n\n{USAGE}").into());
    }
    if trace && use_cuda {
        return Err("--trace runs the CPU pipeline and cannot be combined with --cuda".into());
    }
    #[cfg(not(feature = "cuda"))]
    if use_cuda {
        return Err("CUDA support requires building with cargo oxide and --features cuda".into());
    }
    let path = resolve_scene(Path::new(&path))?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config_path = root.join("config.toml");
    let config = Config::load_scene(&config_path, &scene_name(&path).to_string_lossy())
        .map_err(|error| format!("Failed to load '{}': {error}", config_path.display()))?;
    if config.bounce && !use_cuda {
        return Err("bounce = true in config.toml requires --cuda".into());
    }
    let (render_width, render_height) = config.render_size()?;
    let threads = std::thread::available_parallelism()?.get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let started = Instant::now();
    let asset = GltfAsset::load(&path)
        .map_err(|error| format!("Failed to load '{}': {error}", path.display()))?;
    let loading_time = started.elapsed();
    let started = Instant::now();
    let prepared = prepare_scene(&asset)
        .map_err(|error| format!("Failed to prepare '{}': {error}", path.display()))?;
    let preparation_time = started.elapsed();
    let started = Instant::now();
    #[cfg(feature = "cuda")]
    let mut resident = use_cuda
        .then(|| {
            let device = rasterizer::cuda::CudaDevice::new(0)?;
            Ok::<_, io::Error>(Pipeline::new(
                rasterizer::cuda::CudaVertex::with_device(device.clone()),
                rasterizer::cuda::CudaRasterization::with_device(device.clone()),
                rasterizer::cuda::CudaFragment::with_device(device),
            ))
        })
        .transpose()?;
    let trace_path = root
        .join("output")
        .join(scene_name(&path))
        .with_extension("trace");
    let mut cpu_fragment = cpu::CpuFragment;
    let mut trace_fragment = TraceFragment::new(render_width, &trace_path);
    let fragment: &mut dyn FragmentStage = if trace {
        &mut trace_fragment
    } else {
        &mut cpu_fragment
    };
    let mut pipeline = Pipeline::new(cpu::CpuVertex, cpu::CpuRasterization, fragment);
    let backend_setup_time = started.elapsed();
    let started = Instant::now();
    let camera_matrix = config.camera_matrix()?;
    let frame_setup_time = started.elapsed();
    let transformed = if use_cuda {
        None
    } else {
        Some(
            pipeline
                .transform(&prepared, camera_matrix)
                .map_err(|error| format!("Failed to transform '{}': {error}", path.display()))?,
        )
    };
    let output_dir = root.join("output");
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("Failed to create '{}': {error}", output_dir.display()))?;
    let started = Instant::now();
    let (materials, texels) = load_materials(&asset, config.sampling)?;
    let material_time = started.elapsed();
    eprintln!("Preparing direct-light shadows...");
    let started = Instant::now();
    let source_lights = collect_lights(&prepared, &materials, &texels, &config.light)?;
    let unshadowed = if camera_light {
        let direction = Vec3::from(config.camera.position) - Vec3::from(config.camera.target);
        vec![Light::Directional(DirectionalLight::new(
            direction.to_array(),
            [1.0; 3],
            1.0,
        )?)]
    } else {
        Vec::new()
    };
    let (mut lights, mut maps, mut depths) = (Vec::new(), Vec::new(), Vec::new());
    if !use_cuda {
        (lights, maps) = prepare_lights(
            source_lights.clone(),
            &prepared.world_positions()?,
            &config.shadow,
        )?;
        depths = pipeline.render_shadows(&prepared, &materials, &maps)?;
        lights.extend(unshadowed.iter().map(|&light| RenderLight {
            light,
            shadow: None,
        }));
    }
    #[cfg(feature = "cuda")]
    if let Some(pipeline) = &mut resident {
        pipeline.prepare_cuda_lights(
            &prepared,
            &materials,
            &texels,
            source_lights,
            &config.shadow,
            &unshadowed,
            config.bounce,
        )?;
    }
    let shadow_time = started.elapsed();
    let started = Instant::now();
    let rendered = if let Some(transformed) = &transformed {
        pipeline.render_transformed(
            transformed,
            &materials,
            render_width,
            render_height,
            Shading {
                texels: &texels,
                lights: &lights,
                maps: &maps,
                depths: &depths,
                camera_position: config.camera.position.into(),
            },
        )?
    } else {
        #[cfg(feature = "cuda")]
        {
            resident.as_mut().unwrap().render_cuda(
                &prepared,
                camera_matrix,
                &materials,
                render_width,
                render_height,
                config.camera.position.into(),
            )?
        }
        #[cfg(not(feature = "cuda"))]
        {
            unreachable!()
        }
    };
    let render_time = started.elapsed();
    let output_path = output_dir.join(scene_name(&path)).with_extension("png");
    let started = Instant::now();
    let image = rendered
        .framebuffer
        .to_image(&config.output, config.supersampling)
        .map_err(|error| format!("Failed to postprocess '{}': {error}", path.display()))?;
    let postprocessing_time = started.elapsed();
    let started = Instant::now();
    image
        .save(&output_path)
        .map_err(|error| format!("Failed to save '{}': {error}", output_path.display()))?;
    let output_time = started.elapsed();
    let hits_path = output_path.with_extension("hits");
    #[cfg(feature = "cuda")]
    if config.bounce {
        // Render width and height as u32, then primary and hit material per pixel.
        let hits = resident.as_ref().unwrap().bounce_materials()?;
        let mut bytes = Vec::with_capacity(8 + 4 * hits.len());
        for value in [render_width, render_height].into_iter().chain(hits) {
            bytes.extend(value.to_le_bytes());
        }
        fs::write(&hits_path, bytes)
            .map_err(|error| format!("Failed to save '{}': {error}", hits_path.display()))?;
    }

    let document = &asset.document;
    let primitives: usize = document.meshes().map(|mesh| mesh.primitives().len()).sum();
    println!("Loaded {}", path.display());
    println!("CPU workers: {threads}");
    println!("Backend: {}", if use_cuda { "CUDA" } else { "CPU" });
    println!("Resolution: {}x{}", config.width, config.height);
    println!(
        "Supersampling: {}x per axis ({} samples/pixel); render resolution: {render_width}x{render_height}",
        config.supersampling,
        u64::from(config.supersampling).pow(2)
    );
    println!(
        "Camera (config.toml): position {:?}, target {:?}",
        config.camera.position, config.camera.target
    );
    println!("Scenes: {}", document.scenes().len());
    if let Some(scene) = document.default_scene() {
        println!(
            "Default scene: {} ({})",
            scene.index(),
            scene.name().unwrap_or("unnamed")
        );
    } else {
        println!("Default scene: none");
    }
    println!("Nodes: {}", document.nodes().len());
    println!(
        "Meshes: {} ({primitives} primitives)",
        document.meshes().len()
    );
    println!("Materials: {}", document.materials().len());
    println!("Textures: {}", document.textures().len());
    println!("Images decoded: {}", asset.images.len());
    println!("Buffers loaded: {}", asset.buffers.len());
    println!("Cameras: {}", document.cameras().len());
    println!("Animations: {}", document.animations().len());
    let triangles: usize = prepared.primitives.iter().map(|p| p.triangles.len()).sum();
    let stats = pipeline.stats();
    #[cfg(feature = "cuda")]
    let stats = resident.as_ref().map_or(stats, Pipeline::stats);
    println!("Triangles loaded (selected scene, before clipping): {triangles}");
    println!("Vertices shaded: {}", stats.vertices_shaded);
    println!("Surviving fragments shaded: {}", stats.fragments_shaded);
    match rendered.screen_vertices {
        Some(vertices) => println!(
            "After clipping: {vertices} vertices, {} triangles",
            rendered.screen_triangles
        ),
        None => println!("After clipping: {} triangles", rendered.screen_triangles),
    }
    println!(
        "Covered fragments (before depth testing): {}",
        rendered.covered_fragments
    );
    println!("Timings (static scene with camera changes):");
    if use_cuda {
        println!(
            "  CUDA stage times use events; remaining wall time includes uploads, readback and host setup."
        );
    }
    for (category, timings) in [
        (
            "one-time cost",
            [
                ("Backend setup", backend_setup_time),
                ("glTF loading", loading_time),
                ("Geometry preparation", preparation_time),
                ("Material preparation", material_time),
                ("Shadow preparation", shadow_time),
            ]
            .as_slice(),
        ),
        (
            "per-render cost",
            [
                ("Vertex shading", stats.vertex_time),
                ("Rasterization", stats.rasterization_time),
                ("Fragment shading", stats.fragment_time),
                ("Diffuse bounce", stats.bounce_time),
                (
                    "Render setup + color writeback",
                    (frame_setup_time + render_time).saturating_sub(
                        stats.rasterization_time
                            + stats.fragment_time
                            + stats.bounce_time
                            + if use_cuda {
                                stats.vertex_time
                            } else {
                                Duration::ZERO
                            },
                    ),
                ),
                ("Postprocessing", postprocessing_time),
            ]
            .as_slice(),
        ),
    ] {
        println!("  {category}:");
        let mut total = Duration::ZERO;
        for &(name, duration) in timings {
            println!("    {name}: {:.3} ms", duration.as_secs_f32() * 1000.0);
            total += duration;
        }
        println!(
            "    Total {category}: {:.3} ms",
            total.as_secs_f32() * 1000.0
        );
    }
    println!(
        "  PNG encoding + write (excluded from totals): {:.3} ms",
        output_time.as_secs_f32() * 1000.0
    );
    println!("Wrote {}", output_path.display());
    if trace {
        println!("Wrote {}", trace_path.display());
    }
    if config.bounce {
        println!("Wrote {}", hits_path.display());
        println!("Lighting: direct diffuse + specular, one diffuse bounce");
    } else {
        println!("Lighting: direct diffuse + specular");
    }
    if camera_light {
        println!("Camera light: white, unshadowed");
    }

    let extensions: Vec<_> = document.extensions_used().collect();
    if !extensions.is_empty() {
        println!("Declared extensions: {}", extensions.join(", "));
        println!("Extension metadata is retained; extension behavior is not implemented.");
    }
    Ok(())
}
