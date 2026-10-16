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
    lighting::{DirectionalLight, Light, Lighting, RenderLight, collect_lights},
    material::load_materials,
    pipeline::Pipeline,
    shadow::prepare_lights,
};

const USAGE: &str = "Usage: rasterizer <scene-name|path.gltf|path.glb> [--camera-light]\n\nWithout arguments, list bundled scenes.\n--camera-light: add a white, unshadowed directional light aligned with the camera (off by default).\nRender: cargo run --release -- living-room";

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
    let mut positional = Vec::new();
    for arg in env::args_os().skip(1) {
        if arg == "--camera-light" {
            camera_light = true;
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
        if camera_light {
            return Err(format!("Expected a scene.\n\n{USAGE}").into());
        }
        list_scenes().map_err(|error| format!("Failed to list scenes: {error}"))?;
        return Ok(());
    };

    if args.next().is_some() {
        return Err(format!("Expected one scene name or asset path.\n\n{USAGE}").into());
    }
    let path = resolve_scene(Path::new(&path))?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config_path = root.join("config.toml");
    let config = Config::load_scene(&config_path, &scene_name(&path).to_string_lossy())
        .map_err(|error| format!("Failed to load '{}': {error}", config_path.display()))?;
    let (render_width, render_height) = config.render_size()?;
    let started = Instant::now();
    let asset = GltfAsset::load(&path)
        .map_err(|error| format!("Failed to load '{}': {error}", path.display()))?;
    let loading_time = started.elapsed();
    let started = Instant::now();
    let prepared = prepare_scene(&asset)
        .map_err(|error| format!("Failed to prepare '{}': {error}", path.display()))?;
    let preparation_time = started.elapsed();
    let mut pipeline = Pipeline::new(cpu::CpuVertex, cpu::CpuRasterization, cpu::CpuFragment);
    let started = Instant::now();
    let camera_matrix = config.camera_matrix()?;
    let mut frame_setup_time = started.elapsed();
    let transformed = pipeline
        .transform(&prepared, camera_matrix)
        .map_err(|error| format!("Failed to transform '{}': {error}", path.display()))?;
    let output_dir = root.join("output");
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("Failed to create '{}': {error}", output_dir.display()))?;
    let started = Instant::now();
    let materials = load_materials(&asset, config.sampling)?;
    let material_time = started.elapsed();
    eprintln!("Preparing direct-light shadows...");
    let started = Instant::now();
    let mut lights = prepare_lights(
        collect_lights(&transformed, &materials, &config.light)?,
        &transformed,
        &materials,
        &config.shadow,
    )?;
    let shadow_time = started.elapsed();
    let started = Instant::now();
    if camera_light {
        let direction = Vec3::from(config.camera.position) - Vec3::from(config.camera.target);
        lights.push(RenderLight {
            light: Light::Directional(DirectionalLight::new(direction.to_array(), [1.0; 3], 1.0)?),
            shadow: None,
        });
    }
    frame_setup_time += started.elapsed();
    let started = Instant::now();
    let rendered = pipeline
        .render_transformed(
            &transformed,
            &materials,
            render_width,
            render_height,
            Lighting {
                lights: &lights,
                camera_position: config.camera.position.into(),
            },
        )
        .map_err(|error| format!("Failed to render '{}': {error}", path.display()))?;
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

    let document = &asset.document;
    let primitives: usize = document.meshes().map(|mesh| mesh.primitives().len()).sum();
    println!("Loaded {}", path.display());
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
    println!("Triangles loaded (selected scene, before clipping): {triangles}");
    println!("Vertices shaded: {}", stats.vertices_shaded);
    println!("Surviving fragments shaded: {}", stats.fragments_shaded);
    println!(
        "After clipping: {} vertices, {} triangles",
        rendered.screen_vertices, rendered.screen_triangles
    );
    println!(
        "Covered fragments (before depth testing): {}",
        rendered.covered_fragments
    );
    println!("Timings (wall clock, static scene with camera changes):");
    for (category, timings) in [
        (
            "one-time cost",
            [
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
                (
                    "Render setup + color writeback",
                    frame_setup_time + render_time - stats.rasterization_time - stats.fragment_time,
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
    println!("Lighting: direct diffuse + specular");
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
