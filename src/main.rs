use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

use rasterizer::{
    config::Config,
    cuda::{BOUNCE_RAYS, BOUNCES, Renderer},
    geometry::{GltfAsset, prepare_scene},
    material::load_materials,
    trace::{self, TRACE},
};

const USAGE: &str = "Usage: rasterizer <scene-name|path.gltf|path.glb>\n\nWithout arguments, list bundled scenes.\nRender: ./run.sh living-room";

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
    let mut positional = Vec::new();
    for arg in env::args_os().skip(1) {
        if arg == "--help" || arg == "-h" {
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
    let started = Instant::now();
    let mut renderer = Renderer::new(0)?;
    let backend_setup_time = started.elapsed();
    let started = Instant::now();
    let camera_matrix = config.camera_matrix()?;
    let frame_setup_time = started.elapsed();
    let output_dir = root.join("output");
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("Failed to create '{}': {error}", output_dir.display()))?;
    let started = Instant::now();
    let ntc = config.ntc.then(|| path.with_file_name("ntc"));
    let (materials, texels) = load_materials(&asset, config.sampling, ntc.as_deref())?;
    let material_time = started.elapsed();
    eprintln!("Preparing direct-light shadows...");
    let started = Instant::now();
    renderer.prepare(
        &prepared,
        &materials,
        &texels,
        &config.light,
        &config.shadow,
        config.bounce,
    )?;
    let shadow_time = started.elapsed();
    let started = Instant::now();
    let rendered = renderer.render(
        &prepared,
        camera_matrix,
        render_width,
        render_height,
        config.camera.position.into(),
    )?;
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
    if config.bounce {
        // Render width, height, rays per pixel and bounces per ray as u32, then per pixel the
        // primary material and each ray's hit materials by depth.
        let hits = renderer.bounce_materials()?;
        let mut bytes = Vec::with_capacity(16 + 4 * hits.len());
        for value in [render_width, render_height, BOUNCE_RAYS, BOUNCES]
            .into_iter()
            .chain(hits)
        {
            bytes.extend(value.to_le_bytes());
        }
        fs::write(&hits_path, bytes)
            .map_err(|error| format!("Failed to save '{}': {error}", hits_path.display()))?;
    }
    let trace_path = output_path.with_extension("trace");
    if TRACE {
        trace::write(
            &trace_path,
            render_width,
            render_height,
            &renderer.trace_lanes()?,
            &materials,
            config.sampling,
            config.bounce,
        )
        .map_err(|error| format!("Failed to save '{}': {error}", trace_path.display()))?;
    }

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
    println!(
        "Meshes: {} ({primitives} primitives)",
        document.meshes().len()
    );
    println!(
        "Materials: {} ({} neural)",
        document.materials().len(),
        materials.iter().filter(|m| m.neural.is_some()).count()
    );
    println!("Textures: {}", document.textures().len());
    let triangles: usize = prepared.primitives.iter().map(|p| p.triangles.len()).sum();
    let stats = renderer.stats();
    println!("Triangles loaded (selected scene, before clipping): {triangles}");
    println!("Vertices shaded: {}", stats.vertices_shaded);
    println!("Surviving fragments shaded: {}", stats.fragments_shaded);
    println!("After clipping: {} triangles", rendered.screen_triangles);
    println!(
        "Covered fragments (before depth testing): {}",
        rendered.covered_fragments
    );
    println!("Timings (static scene with camera changes):");
    println!(
        "  CUDA stage times use events; remaining wall time includes uploads, readback and host setup."
    );
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
                        stats.vertex_time
                            + stats.rasterization_time
                            + stats.fragment_time
                            + stats.bounce_time,
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
    if config.bounce {
        println!("Wrote {}", hits_path.display());
        println!("Lighting: direct diffuse + specular, diffuse bounces");
        println!("Bounce rays per pixel: {BOUNCE_RAYS}, bounces per ray: {BOUNCES}");
    } else {
        println!("Lighting: direct diffuse + specular");
    }
    if TRACE {
        println!("Wrote {}", trace_path.display());
    }
    Ok(())
}
