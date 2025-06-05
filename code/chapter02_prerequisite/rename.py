import os

folder_path = os.path.expanduser("~/Desktop/LLM_Interview")

rename_mapping = {
    # 模块一
    "基础面": "阶段1-基础认知",
    "幻觉面": "阶段1-基础认知",
    "显存问题面": "阶段1-基础认知",
    "Tokenizer": "阶段1-基础认知",

    # 模块二
    "训练面": "阶段2-训练与调优",
    "增量预训练": "阶段2-训练与调优",
    "强化学习面": "阶段2-训练与调优",
    "强化学习—— PPO 面": "阶段2-训练与调优",
    "RLHF": "阶段2-训练与调优",
    "蒸馏面": "阶段2-训练与调优",
    "微调面": "阶段2-训练与调优",
    "PRFT": "阶段2-训练与调优",
    "LoRA": "阶段2-训练与调优",
    "Adapter": "阶段2-训练与调优",
    "适配架构": "阶段2-训练与调优",

    # 模块三
    "推理加速": "阶段3-推理与部署",
    "推理面": "阶段3-推理与部署",

    # 模块四
    "RAG 检索增强": "阶段4-RAG与Agent",
    "agent 面": "阶段4-RAG与Agent",
    "langchain 面": "阶段4-RAG与Agent",
    "Prompt": "阶段4-RAG与Agent",
    "向量库": "阶段4-RAG与Agent",

    # 模块五
    "评测面": "阶段5-评估与面试",
    "多模态常见面试题": "阶段5-评估与面试",
    "投招面试题": "阶段5-评估与面试",
    "面试题答卷": "阶段5-评估与面试",
}

execute = True

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        matched = False
        for keyword, prefix in rename_mapping.items():
            if keyword in filename:
                new_filename = f"{prefix}-{filename}"
                src = os.path.join(folder_path, filename)
                dst = os.path.join(folder_path, new_filename)
                print(f"✅ 重命名: {filename} -> {new_filename}")
                if execute:
                    os.rename(src, dst)
                matched = True
                break
        if not matched:
            print(f"⚠️ 未匹配规则: {filename}")

print("✅ 重命名完成")
