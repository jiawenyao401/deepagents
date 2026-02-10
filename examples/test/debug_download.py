from deepagents.backends.filesystem import FilesystemBackend

fs_backend = FilesystemBackend(root_dir="D:\\test", virtual_mode=True)

print("=== Testing download_files ===")
paths = ["/skill/skills/skills/pptx/SKILL.md", "/skill/skills/skills/docx/SKILL.md"]
responses = fs_backend.download_files(paths)

for path, response in zip(paths, responses):
    print(f"\nPath: {path}")
    print(f"Error: {response.error}")
    if response.content:
        print(f"Content length: {len(response.content)} bytes")
        print(f"First 200 chars: {response.content[:200]}")
