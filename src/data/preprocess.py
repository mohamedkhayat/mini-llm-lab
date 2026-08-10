import os
import re
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Normalize whitespace: collapse multiple blank lines, strip per-line."""
    text = re.sub(r"\n\s*\n", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def combine_files(
    input_dir: str,
    output_dir: str,
    max_size_mb: int = 500,
    separator: str = "\n\n",
    fallback_encoding: str = "latin1",
) -> int:
    """Concatenate text files from ``input_dir`` into chunks in ``output_dir``.

    Each output file is at most ``max_size_mb`` megabytes.
    Returns the number of output files written.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(
        fp
        for root, _, files in os.walk(input_dir)
        for f in files
        if f.endswith((".txt", ".txt.utf8"))
        for fp in [os.path.join(root, f)]
    )

    current_parts: list[str] = []
    current_size = 0
    counter = 1

    for filepath in tqdm(all_files, desc="Combining files"):
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding=fallback_encoding) as fh:
                content = fh.read()

        estimated = len(content.encode("utf-8"))
        if current_size + estimated > max_size_mb * 1024 * 1024 and current_parts:
            out_path = os.path.join(output_dir, f"combined_{counter}.txt")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(separator.join(current_parts))
            counter += 1
            current_parts = [content]
            current_size = estimated
        else:
            current_parts.append(content)
            current_size += estimated

    if current_parts:
        out_path = os.path.join(output_dir, f"combined_{counter}.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(separator.join(current_parts))
        counter += 1

    return counter - 1
