use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use rasterizer::{
    asset::GltfAsset,
    backend::cpu,
    bake,
    clip::clip_scene,
    config::Config,
    lighting::collect_lights,
    lightmap::{BakedLighting, cache_key},
    material::load_materials,
    vertex::transform_scene,
    viewport::project_scene,
};

const USAGE: &str = "Usage: rasterizer [bake] <scene-name|path.gltf|path.glb>\n\nWithout arguments, list bundled scenes.\nBake: cargo run --release -- bake kitchen\nRender: cargo run --release -- kitchen";

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
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        list_scenes().map_err(|error| format!("Failed to list scenes: {error}"))?;
        return Ok(());
    };

    let baking = path == "bake";
    let path = if baking {
        args.next()
            .ok_or_else(|| format!("Expected a scene to bake.\n\n{USAGE}"))?
    } else {
        path
    };
    if args.next().is_some() {
        return Err(format!("Expected one scene name or asset path.\n\n{USAGE}").into());
    }
    if path == "--help" || path == "-h" {
        println!("{USAGE}");
        return Ok(());
    }

    let path = resolve_scene(Path::new(&path))?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config_path = root.join("config.toml");
    let config = Config::load(&config_path)
        .map_err(|error| format!("Failed to load '{}': {error}", config_path.display()))?;
    let asset = GltfAsset::load(&path)
        .map_err(|error| format!("Failed to load '{}': {error}", path.display()))?;
    let mut transformed = transform_scene(&asset, &config)
        .map_err(|error| format!("Failed to transform '{}': {error}", path.display()))?;
    let output_dir = root.join("output");
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("Failed to create '{}': {error}", output_dir.display()))?;
    let cache_path = output_dir
        .join(scene_name(&path))
        .with_extension("lightmap");
    let cached = if baking {
        None
    } else {
        Some(
            BakedLighting::load(&cache_path, cache_key(&asset, &config)?).map_err(|error| {
                format!(
                    "Failed to load '{}': {error}\nRun: cargo run --release -- bake {}",
                    cache_path.display(),
                    path.display()
                )
            })?,
        )
    };
    let materials = load_materials(&asset)?;
    eprintln!("Preparing direct-light shadows...");
    let lights = cpu::prepare_lights(
        collect_lights(&transformed, &materials, &config.light)?,
        &transformed,
        &materials,
        &config.shadow,
    )?;
    let lighting = if let Some(lighting) = cached {
        lighting.apply(&mut transformed)?;
        lighting
    } else {
        let lighting = bake::bake(&mut transformed, &asset, &materials, &lights, &config)
            .map_err(|error| format!("Failed to bake '{}': {error}", path.display()))?;
        lighting
            .save(&cache_path)
            .map_err(|error| format!("Failed to save '{}': {error}", cache_path.display()))?;
        println!("Baked {}", cache_path.display());
        lighting
    };
    let clipped = clip_scene(&transformed)
        .map_err(|error| format!("Failed to clip '{}': {error}", path.display()))?;
    let screen = project_scene(&clipped, config.width, config.height)
        .map_err(|error| format!("Failed to project '{}': {error}", path.display()))?;
    let rendered = cpu::render(
        &screen,
        &materials,
        config.width,
        config.height,
        cpu::Lighting {
            lightmap: Some(&lighting.lightmap),
            lights: &lights,
            camera_position: config.camera.position.into(),
        },
    )
    .map_err(|error| format!("Failed to render '{}': {error}", path.display()))?;
    let output_path = output_dir.join(scene_name(&path)).with_extension("png");
    rendered
        .framebuffer
        .save(&output_path)
        .map_err(|error| format!("Failed to save '{}': {error}", output_path.display()))?;

    let document = &asset.document;
    let primitives: usize = document.meshes().map(|mesh| mesh.primitives().len()).sum();
    println!("Loaded {}", path.display());
    println!("Resolution: {}x{}", config.width, config.height);
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
    let vertices: usize = transformed
        .primitives
        .iter()
        .map(|p| p.vertices.len())
        .sum();
    let triangles: usize = transformed
        .primitives
        .iter()
        .map(|p| p.triangles.len())
        .sum();
    println!(
        "Clip-space scene {}: {} primitives, {vertices} vertices, {triangles} triangles",
        transformed.scene_index,
        transformed.primitives.len()
    );
    let clipped_triangles: usize = clipped.primitives.iter().map(|p| p.triangles.len()).sum();
    println!("After clipping: {clipped_triangles} triangles");
    let screen_vertices: usize = screen.primitives.iter().map(|p| p.vertices.len()).sum();
    let screen_triangles: usize = screen.primitives.iter().map(|p| p.triangles.len()).sum();
    println!("Pixel-space: {screen_vertices} vertices, {screen_triangles} triangles");
    println!(
        "Covered fragments (before depth testing): {}",
        rendered.covered_fragments
    );
    println!("Wrote {}", output_path.display());
    println!(
        "Lighting: {}x{} baked diffuse (direct + one bounce), plus shadowed runtime direct specular",
        lighting.lightmap.width, lighting.lightmap.height
    );

    let extensions: Vec<_> = document.extensions_used().collect();
    if !extensions.is_empty() {
        println!("Declared extensions: {}", extensions.join(", "));
        println!("Extension metadata is retained; extension behavior is not implemented.");
    }
    Ok(())
}
