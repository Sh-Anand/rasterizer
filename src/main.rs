use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use rasterizer::asset::GltfAsset;

const USAGE: &str = "Usage: rasterizer [scene-name|path.gltf|path.glb]\n\nWithout arguments, list bundled scenes. Example: cargo run -- kitchen";

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
    let mut args = env::args_os().skip(1);
    let Some(path) = args.next() else {
        return match list_scenes() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("Failed to list scenes: {error}");
                ExitCode::FAILURE
            }
        };
    };

    if args.next().is_some() {
        eprintln!("Expected one scene name or asset path.\n\n{USAGE}");
        return ExitCode::FAILURE;
    }
    if path == "--help" || path == "-h" {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let path = match resolve_scene(Path::new(&path)) {
        Ok(path) => path,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let asset = match GltfAsset::load(&path) {
        Ok(asset) => asset,
        Err(error) => {
            eprintln!("Failed to load '{}': {error}", path.display());
            return ExitCode::FAILURE;
        }
    };

    let document = &asset.document;
    let primitives: usize = document.meshes().map(|mesh| mesh.primitives().len()).sum();
    println!("Loaded {}", path.display());
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

    let extensions: Vec<_> = document.extensions_used().collect();
    if !extensions.is_empty() {
        println!("Declared extensions: {}", extensions.join(", "));
        println!("Extension metadata is retained; extension behavior is not implemented.");
    }
    ExitCode::SUCCESS
}
