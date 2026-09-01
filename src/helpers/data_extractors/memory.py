import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import tempfile

class Hash:
    def __init__(
        self,
        confidence: Optional[float] = None,
        cache_folder: Path = Path("./cache"),
    ) -> None:
        self.confidence = confidence
        self.cache_folder = cache_folder
        self.format = "parquet"
        self.create_cache_folder()  # Create folder first
        self._check_parquet_support()  # Then check parquet support

    def _check_parquet_support(self) -> None:
        """Check if parquet is available, raise error if not."""
        try:
            # Test if parquet is available using a temporary file
            test_df = pd.DataFrame({"test": [1]})
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                test_path = Path(tmp.name)
            test_df.to_parquet(test_path)
            test_path.unlink()  # cleanup
        except ImportError as e:
            raise ImportError(
                "Parquet support is required but not available. "
                "Install pyarrow: pip install pyarrow"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Parquet test failed: {e}") from e

    def generate_hash_key(self, **kwargs) -> str:
        """Generate a more robust hash key"""
        # Create a normalized dictionary for consistent hashing
        hash_data = {"confidence": self.confidence, **kwargs}
        # Sort keys for consistent hashing
        normalized_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(normalized_str.encode()).hexdigest()

    def exists(self, hash_key: str) -> bool:
        """Check if cached data exists"""
        path = self._get_cache_path(hash_key)
        return path.exists()

    def load(self, hash_key: str) -> pd.DataFrame:
        """Load cached data"""
        path = self._get_cache_path(hash_key)
        return pd.read_parquet(path)

    def save(self, data: pd.DataFrame, hash_key: str) -> None:
        """Save data to cache"""
        path = self._get_cache_path(hash_key)
        data.to_parquet(path)

    def _get_cache_path(self, hash_key: str) -> Path:
        """Get the full path for a cache file"""
        filename = f"{hash_key}.parquet"
        return self.cache_folder / filename

    def create_cache_folder(self) -> None:
        """Create cache folder if it doesn't exist"""
        self.cache_folder.mkdir(parents=True, exist_ok=True)

    def clear_cache(self) -> None:
        """Clear all cached files"""
        for file in self.cache_folder.glob("*.parquet"):
            file.unlink()

    def cache_info(self) -> Dict[str, Any]:
        """Get information about cached files"""
        files = list(self.cache_folder.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "num_files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "format": "parquet",
        }

    # Backward compatibility methods
    def check_if_analysis_run_before(self, hash_key: str) -> bool:
        """Backward compatibility: Check if cached data exists"""
        return self.exists(hash_key)

    def load_pickled_analysis(self, hash_key: str) -> pd.DataFrame:
        """Backward compatibility: Load cached data (parquet format)"""
        return self.load(hash_key)

    def pickle_current_analysis(
        self, data: pd.DataFrame, hash_key: str
    ) -> None:
        """Backward compatibility: Save data to cache (parquet format)"""
        self.save(data, hash_key)

    def get_path_to_pickle(self, hash_key: str) -> Path:
        """Backward compatibility: Get cache file path"""
        return self._get_cache_path(hash_key)