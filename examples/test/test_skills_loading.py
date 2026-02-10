from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills

# 测试 skill 加载
fs_backend = FilesystemBackend(root_dir="D:\\test", virtual_mode=True)

print("=== Testing PPTX Skill Loading ===\n")

# 测试 pptx skill
pptx_skills = _list_skills(fs_backend, "/skill/skills/skills")
print(f"Found {len(pptx_skills)} skills in /skill/skills/skills directory:")
for skill in pptx_skills:
    print(f"\nName: {skill['name']}")
    print(f"Description: {skill['description'][:100]}...")
    print(f"Path: {skill['path']}")

print("\n=== All Skills Loaded ===")
print(f"Total: {len(pptx_skills)} skills")
