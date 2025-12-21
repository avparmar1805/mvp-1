#!/usr/bin/env python3
"""
Data Product Schema Validator

Validates Data Product specifications against the JSON schema.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
import yaml
from jsonschema import ValidationError
from loguru import logger


class DataProductValidator:
    """Validates Data Product specifications against the JSON schema."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the validator.
        
        Args:
            schema_path: Path to JSON schema file (defaults to data_product_schema.json)
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "data_product_schema.json"
        
        self.schema_path = Path(schema_path)
        
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, "r") as f:
            self.schema = json.load(f)
        
        logger.info(f"Loaded schema from {self.schema_path}")
    
    def validate(self, spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a Data Product specification.
        
        Args:
            spec: Data Product specification dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            jsonschema.validate(instance=spec, schema=self.schema)
            logger.debug("Validation passed")
            return True, []
        except ValidationError as e:
            errors.append(f"Validation error: {e.message}")
            errors.append(f"  Path: {' -> '.join(str(p) for p in e.path)}")
            if e.absolute_path:
                errors.append(f"  Location: {list(e.absolute_path)}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error during validation: {e}")
            return False, errors
    
    def validate_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a Data Product specification file.
        
        Args:
            file_path: Path to YAML or JSON file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, [f"File not found: {file_path}"]
        
        try:
            # Load file based on extension
            if file_path.suffix in [".yaml", ".yml"]:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
                    # Handle YAML files with 'data_product' wrapper
                    if "data_product" in data:
                        spec = data["data_product"]
                    else:
                        spec = data
            elif file_path.suffix == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "data_product" in data:
                        spec = data["data_product"]
                    else:
                        spec = data
            else:
                return False, [f"Unsupported file format: {file_path.suffix}"]
            
            return self.validate(spec)
            
        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {e}"]
        except json.JSONDecodeError as e:
            return False, [f"JSON parsing error: {e}"]
        except Exception as e:
            return False, [f"Error reading file: {e}"]
    
    def validate_directory(self, dir_path: str) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Validate all Data Product specifications in a directory.
        
        Args:
            dir_path: Path to directory containing spec files
            
        Returns:
            Dictionary mapping file paths to (is_valid, errors) tuples
        """
        dir_path = Path(dir_path)
        results = {}
        
        # Find all YAML and JSON files
        for file_path in dir_path.glob("*.yaml"):
            results[str(file_path)] = self.validate_file(str(file_path))
        
        for file_path in dir_path.glob("*.yml"):
            results[str(file_path)] = self.validate_file(str(file_path))
        
        for file_path in dir_path.glob("*.json"):
            results[str(file_path)] = self.validate_file(str(file_path))
        
        return results


def main():
    """CLI for validating Data Product specifications."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Data Product specifications"
    )
    parser.add_argument(
        "file_or_dir",
        type=str,
        help="Path to YAML/JSON file or directory containing specs"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to JSON schema file (default: data_product_schema.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed error messages"
    )
    
    args = parser.parse_args()
    
    validator = DataProductValidator(schema_path=args.schema)
    path = Path(args.file_or_dir)
    
    if path.is_file():
        # Validate single file
        is_valid, errors = validator.validate_file(str(path))
        
        if is_valid:
            print(f"✅ {path.name} is valid")
            sys.exit(0)
        else:
            print(f"❌ {path.name} is invalid:")
            for error in errors:
                print(f"  {error}")
            sys.exit(1)
    
    elif path.is_dir():
        # Validate directory
        results = validator.validate_directory(str(path))
        
        valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
        total_count = len(results)
        
        print(f"\nValidation Results: {valid_count}/{total_count} valid\n")
        
        for file_path, (is_valid, errors) in results.items():
            status = "✅" if is_valid else "❌"
            print(f"{status} {Path(file_path).name}")
            
            if not is_valid and args.verbose:
                for error in errors:
                    print(f"    {error}")
        
        sys.exit(0 if valid_count == total_count else 1)
    
    else:
        print(f"❌ Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

