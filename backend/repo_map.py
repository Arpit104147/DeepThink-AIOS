import os
import ast

class RepoMapGenerator:
    """
    Generates a highly compressed Abstract Syntax Tree (AST) map of a Python repository.
    This allows an LLM to understand the architecture of massive codebases (like SWE-bench repos)
    using only a tiny fraction of the context window.
    """
    def __init__(self, root_dir, ignore_dirs=None):
        self.root_dir = os.path.abspath(root_dir)
        self.ignore_dirs = set(ignore_dirs or ['.git', '__pycache__', 'venv', 'env', 'node_modules', 'dist', 'build', '.mypy_cache', '.pytest_cache'])

    def _parse_file(self, file_path):
        """Parse a single Python file and return its structural signature."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception:
            # If the file is not valid Python or unreadable, return empty
            return []

        lines = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}:")
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef) or isinstance(sub_node, ast.AsyncFunctionDef):
                        # Extract arguments safely
                        args = [arg.arg for arg in sub_node.args.args]
                        args_str = ", ".join(args)
                        lines.append(f"    def {sub_node.name}({args_str})")
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                args = [arg.arg for arg in node.args.args]
                args_str = ", ".join(args)
                lines.append(f"def {node.name}({args_str})")
                
        return lines

    def generate_map(self, prompt=None, max_chars=24000):
        """Walks the repository, scores files based on relevance to the prompt,
        and generates a prioritized AST map up to max_chars to prevent context window blowout.
        """
        file_entries = []
        
        # 1. Collect all Python files
        for root, dirs, files in os.walk(self.root_dir):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs and not d.startswith('.')]
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.root_dir)
                file_entries.append((file_path, rel_path))
                
        # 2. Score files based on relevance to prompt
        scored_entries = []
        if prompt:
            prompt_lower = prompt.lower()
            for file_path, rel_path in file_entries:
                score = 0
                parts = rel_path.lower().split(os.sep)
                filename = parts[-1]
                
                # Check if exact filename is mentioned (high priority)
                if filename in prompt_lower:
                    score += 100
                # Check if parts of the path are mentioned (medium priority)
                for part in parts[:-1]:
                    if len(part) > 3 and part in prompt_lower:
                        score += 30
                        
                scored_entries.append((score, file_path, rel_path))
        else:
            # Uniform score if no prompt
            scored_entries = [(0, fp, rp) for fp, rp in file_entries]
            
        # 3. Sort by score (descending) and then alphabetically by path for determinism
        scored_entries.sort(key=lambda x: (-x[0], x[2]))
        
        # 4. Generate signatures up to max_chars
        map_output = []
        current_len = 0
        truncated = False
        
        for score, file_path, rel_path in scored_entries:
            signatures = self._parse_file(file_path)
            if signatures:
                entry_header = f"# {rel_path}"
                entry_body = "\n".join(signatures) + "\n\n"
                entry_str = f"{entry_header}\n{entry_body}"
                
                if current_len + len(entry_str) > max_chars:
                    truncated = True
                    break
                    
                map_output.append(entry_str)
                current_len += len(entry_str)
                
        if truncated:
            map_output.append("# ... [other repository files truncated to fit context limits]")
            
        return "".join(map_output).strip()

if __name__ == "__main__":
    # Test execution: Map the backend directory itself
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    repo_map = RepoMapGenerator(backend_dir).generate_map()
    print("=== AST REPO MAP FOR BACKEND ===")
    print(repo_map)
    print("================================")
    print(f"Total Tokens (approx): {len(repo_map) // 4}")
