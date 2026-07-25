use std::path::PathBuf;

pub struct Config {
    pub db_path: PathBuf,
    pub model_dir: PathBuf,
}

impl Config {
    /// Resolve config from explicit values, falling back to env vars, then .env, then defaults.
    /// `cli_db` and `cli_model` represent values passed via CLI flags (None if not passed).
    pub fn resolve(cli_db: Option<PathBuf>, cli_model: Option<PathBuf>) -> anyhow::Result<Self> {
        // Load .env silently (ignore if missing)
        let _ = dotenvy::dotenv();

        let db_path = cli_db
            .or_else(|| std::env::var("BSEARCH_DB_PATH").ok().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("bsearch.db"));

        // Use ~/.cache/bsearch/ to match the Python export-model command,
        // which uses the XDG convention rather than the macOS Library/Caches path.
        let default_model_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cache")
            .join("bsearch")
            .join("all-MiniLM-L6-v2");

        let model_dir = cli_model
            .or_else(|| std::env::var("BSEARCH_MODEL_DIR").ok().map(PathBuf::from))
            .unwrap_or(default_model_dir);

        Ok(Config { db_path, model_dir })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_defaults_when_no_env() {
        unsafe {
            std::env::remove_var("BSEARCH_DB_PATH");
            std::env::remove_var("BSEARCH_MODEL_DIR");
        }

        let config = Config::resolve(None, None).unwrap();
        assert_eq!(config.db_path, PathBuf::from("bsearch.db"));
        assert!(config.model_dir.ends_with("bsearch/all-MiniLM-L6-v2"));
    }

    #[test]
    #[serial]
    fn test_cli_overrides_env() {
        unsafe {
            std::env::set_var("BSEARCH_DB_PATH", "/env/path.db");
        }
        let config = Config::resolve(Some(PathBuf::from("/cli/path.db")), None).unwrap();
        assert_eq!(config.db_path, PathBuf::from("/cli/path.db"));
        unsafe {
            std::env::remove_var("BSEARCH_DB_PATH");
        }
    }

    #[test]
    #[serial]
    fn test_env_var_used_when_no_cli() {
        unsafe {
            std::env::set_var("BSEARCH_DB_PATH", "/env/path.db");
        }
        let config = Config::resolve(None, None).unwrap();
        assert_eq!(config.db_path, PathBuf::from("/env/path.db"));
        unsafe {
            std::env::remove_var("BSEARCH_DB_PATH");
        }
    }
}
