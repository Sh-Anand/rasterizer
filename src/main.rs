use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use glam::Vec3;
use rasterizer::{
    config::{Config, OutputConfig},
    cuda::{BOUNCE_RAYS, BOUNCES, Renderer},
    geometry::{GltfAsset, prepare_scene},
    material::{Network, load_materials},
    trace::{self, TRACE},
};

/// The glTF file in each folder of assets/scenes, named by its folder.
fn scenes() -> io::Result<Vec<PathBuf>> {
    let mut scenes = Vec::new();
    for folder in fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/scenes"))? {
        let folder = folder?.path();
        if !folder.is_dir() {
            continue;
        }
        for file in fs::read_dir(folder)? {
            let path = file?.path();
            if path
                .extension()
                .is_some_and(|ext| ext == "gltf" || ext == "glb")
            {
                scenes.push(path);
            }
        }
    }
    scenes.sort();
    Ok(scenes)
}

fn scene_name(path: &Path) -> String {
    path.parent()
        .and_then(Path::file_name)
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
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
    let Some(arg) = env::args_os().nth(1) else {
        println!("Scenes:");
        for scene in scenes()? {
            println!("  {}", scene_name(&scene));
        }
        return Ok(());
    };
    // A bundled scene's name, or a glTF path.
    let arg = Path::new(&arg);
    let path = if arg.is_file() {
        arg.to_owned()
    } else {
        scenes()?
            .into_iter()
            .find(|scene| scene_name(scene) == arg.to_string_lossy())
            .ok_or_else(|| format!("Unknown scene '{}'", arg.display()))?
    };
    let scene = scene_name(&path);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = Config::load(root.join("config.toml"), &scene)?;
    let (width, height) = config.render_size();
    let asset = GltfAsset::load(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let prepared = prepare_scene(&asset)?;
    let (materials, texels) = load_materials(
        &asset,
        config.sampling,
        config.neural,
        &path.with_file_name("ntc"),
    )?;
    let mut renderer = Renderer::new(0)?;
    renderer.prepare(
        &prepared,
        &materials,
        &texels,
        &config.light,
        &config.shadow,
        config.bounce,
    )?;
    let color = renderer.render(
        &prepared,
        config.camera_matrix(),
        width,
        height,
        config.camera.position.into(),
    )?;
    let output = root.join("output").join(&scene);
    fs::create_dir_all(root.join("output"))?;
    resolve(&color, width, height, &config.output, config.supersampling)
        .save(output.with_extension("png"))?;
    if TRACE {
        trace::write(
            output.with_extension("trace"),
            width,
            height,
            &renderer.trace_lanes()?,
            &materials,
            config.sampling,
            config.bounce,
        )?;
    }

    let neural = materials.iter().filter(|m| m.network != Network::None);
    println!(
        "{scene}: {width}x{height} samples, {} materials ({} neural)",
        materials.len(),
        neural.count()
    );
    if config.bounce {
        println!("Bounce rays per pixel: {BOUNCE_RAYS}, bounces per ray: {BOUNCES}");
    }
    // Event intervals can include host launch delays.
    let stats = renderer.stats();
    for (stage, time) in [
        ("Vertex shading", stats.vertex_time),
        ("Rasterization", stats.rasterization_time),
        ("Fragment shading", stats.fragment_time),
        ("Diffuse bounce", stats.bounce_time),
    ] {
        println!("    {stage}: {:.3} ms", time.as_secs_f32() * 1000.0);
    }
    println!("Wrote {}", output.with_extension("png").display());
    if TRACE {
        println!("Wrote {}", output.with_extension("trace").display());
    }
    Ok(())
}

/// Box-filters each pixel's supersamples in linear HDR, then applies exposure, per-channel
/// Reinhard tone mapping and sRGB encoding.
fn resolve(
    color: &[Vec3],
    width: u32,
    height: u32,
    output: &OutputConfig,
    supersampling: u32,
) -> image::RgbImage {
    let samples = supersampling as f32 * supersampling as f32;
    image::RgbImage::from_fn(width / supersampling, height / supersampling, |x, y| {
        let mut sum = Vec3::ZERO;
        for sy in y * supersampling..(y + 1) * supersampling {
            let start = sy as usize * width as usize + (x * supersampling) as usize;
            sum += color[start..start + supersampling as usize]
                .iter()
                .copied()
                .sum::<Vec3>();
        }
        let mut color = (sum / samples).max(Vec3::ZERO) * output.exposure;
        if output.tone_mapping {
            color /= Vec3::ONE + color;
        }
        image::Rgb(color.to_array().map(|value| {
            let value = value.clamp(0.0, 1.0);
            let encoded = if value <= 0.0031308 {
                value * 12.92
            } else {
                1.055 * value.powf(1.0 / 2.4) - 0.055
            };
            (encoded * 255.0).round() as u8
        }))
    })
}
