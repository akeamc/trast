use std::path::PathBuf;

use cached_path::Cache;

use crate::Result;

fn ensure_cache_dir() -> std::io::Result<PathBuf> {
    let mut dir = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
    dir.push("trast");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn download(url: impl AsRef<str>) -> Result<PathBuf> {
    let url = url.as_ref();
    let dir = ensure_cache_dir()?;
    let cache = Cache::builder().dir(dir).build()?;

    Ok(cache.cached_path(url)?)
}
