"""utils/cache.py
=============================================================================
Sistema de cache inteligente para resultados intermediários.

Evita reprocessamento desnecessário de dados quando os inputs não mudaram.
Usa hash de arquivos e parâmetros para detectar mudanças.
=============================================================================
"""

import sys
from pathlib import Path

# Adiciona o diretório parent ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import json
import pickle
from typing import Any, Optional, Callable
from functools import wraps
import pandas as pd

from config import CACHE_DIR


def file_hash(file_path: Path) -> str:
    """
    Calcula hash MD5 de um arquivo.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo.

    Returns
    -------
    str
        Hash MD5 do arquivo.
    """
    if not file_path.exists():
        return ""

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def params_hash(**kwargs) -> str:
    """
    Calcula hash de parâmetros.

    Parameters
    ----------
    **kwargs
        Parâmetros arbitrários para gerar hash.

    Returns
    -------
    str
        Hash MD5 dos parâmetros.
    """
    # Serializa parâmetros de forma determinística
    serialized = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


class CacheManager:
    """
    Gerenciador de cache para resultados intermediários.

    Examples
    --------
    >>> cache = CacheManager()
    >>> result = cache.get_or_compute(
    ...     key="preprocessing_conservador",
    ...     compute_fn=lambda: preprocess_data("conservador"),
    ...     dependencies=["data/raw/fundamentals.csv"]
    ... )
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        """
        Inicializa o gerenciador de cache.

        Parameters
        ----------
        cache_dir : Path
            Diretório para armazenar cache.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Carrega metadados do cache."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Salva metadados do cache."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_dependencies_hash(self, dependencies: list) -> str:
        """
        Calcula hash combinado de todos os arquivos de dependência.

        Parameters
        ----------
        dependencies : list
            Lista de paths de arquivos.

        Returns
        -------
        str
            Hash combinado.
        """
        hashes = []
        for dep in dependencies:
            dep_path = Path(dep)
            if dep_path.exists():
                hashes.append(file_hash(dep_path))
        return hashlib.md5("".join(hashes).encode()).hexdigest()

    def get_cache_path(self, key: str, format: str = "pkl") -> Path:
        """
        Retorna o path do arquivo de cache.

        Parameters
        ----------
        key : str
            Chave do cache.
        format : str
            Formato do arquivo (pkl, csv, json, parquet).

        Returns
        -------
        Path
            Path do arquivo de cache.
        """
        return self.cache_dir / f"{key}.{format}"

    def is_valid(self, key: str, dependencies: Optional[list] = None) -> bool:
        """
        Verifica se o cache é válido.

        Parameters
        ----------
        key : str
            Chave do cache.
        dependencies : list, optional
            Lista de arquivos de dependência.

        Returns
        -------
        bool
            True se o cache é válido, False caso contrário.
        """
        if key not in self.metadata:
            return False

        cache_path = self.get_cache_path(key)
        if not cache_path.exists():
            return False

        if dependencies:
            current_deps_hash = self._compute_dependencies_hash(dependencies)
            cached_deps_hash = self.metadata[key].get("dependencies_hash")
            return current_deps_hash == cached_deps_hash

        return True

    def save(
        self,
        key: str,
        data: Any,
        dependencies: Optional[list] = None,
        format: str = "pkl"
    ):
        """
        Salva dados no cache.

        Parameters
        ----------
        key : str
            Chave do cache.
        data : Any
            Dados para cachear.
        dependencies : list, optional
            Lista de arquivos de dependência.
        format : str
            Formato do arquivo (pkl, csv, json, parquet).
        """
        cache_path = self.get_cache_path(key, format)

        # Salva dados no formato apropriado
        if format == "pkl":
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        elif format == "csv" and isinstance(data, pd.DataFrame):
            data.to_csv(cache_path, index=False)
        elif format == "json":
            if isinstance(data, pd.DataFrame):
                data.to_json(cache_path, orient="records", indent=2)
            else:
                with open(cache_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
        elif format == "parquet" and isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path, index=False)
        else:
            raise ValueError(f"Formato não suportado: {format}")

        # Atualiza metadados
        self.metadata[key] = {
            "format": format,
            "dependencies_hash": (
                self._compute_dependencies_hash(dependencies)
                if dependencies else None
            )
        }
        self._save_metadata()

    def load(self, key: str) -> Any:
        """
        Carrega dados do cache.

        Parameters
        ----------
        key : str
            Chave do cache.

        Returns
        -------
        Any
            Dados cacheados.
        """
        if key not in self.metadata:
            raise KeyError(f"Cache key não encontrado: {key}")

        format = self.metadata[key]["format"]
        cache_path = self.get_cache_path(key, format)

        if format == "pkl":
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        elif format == "csv":
            return pd.read_csv(cache_path)
        elif format == "json":
            with open(cache_path, "r") as f:
                data = json.load(f)
                # Tenta converter para DataFrame se parece com tabela
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                return data
        elif format == "parquet":
            return pd.read_parquet(cache_path)
        else:
            raise ValueError(f"Formato não suportado: {format}")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        dependencies: Optional[list] = None,
        format: str = "pkl",
        force: bool = False
    ) -> Any:
        """
        Recupera do cache ou computa se necessário.

        Parameters
        ----------
        key : str
            Chave do cache.
        compute_fn : Callable
            Função para computar o resultado se não estiver em cache.
        dependencies : list, optional
            Lista de arquivos de dependência.
        format : str
            Formato do arquivo.
        force : bool
            Se True, ignora cache e recomputa.

        Returns
        -------
        Any
            Resultado cacheado ou computado.
        """
        if not force and self.is_valid(key, dependencies):
            return self.load(key)

        # Computa resultado
        result = compute_fn()

        # Salva no cache
        self.save(key, result, dependencies, format)

        return result

    def clear(self, key: Optional[str] = None):
        """
        Limpa o cache.

        Parameters
        ----------
        key : str, optional
            Se fornecido, limpa apenas essa chave. Caso contrário, limpa tudo.
        """
        if key:
            cache_path = self.get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
        else:
            # Limpa tudo
            for file in self.cache_dir.glob("*"):
                if file != self.metadata_file:
                    file.unlink()
            self.metadata = {}
            self._save_metadata()


def cached(
    key: str,
    dependencies: Optional[list] = None,
    format: str = "pkl"
):
    """
    Decorator para cachear resultados de funções.

    Parameters
    ----------
    key : str
        Chave do cache.
    dependencies : list, optional
        Lista de arquivos de dependência.
    format : str
        Formato do cache.

    Examples
    --------
    >>> @cached("preprocessing_conservador", dependencies=["data/raw/data.csv"])
    ... def preprocess_conservador():
    ...     # processamento pesado
    ...     return result
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = CacheManager()
            return cache.get_or_compute(
                key=key,
                compute_fn=lambda: func(*args, **kwargs),
                dependencies=dependencies,
                format=format
            )
        return wrapper
    return decorator
