from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC, Sequence
import os
import json

vs = 8192

class MultilingualBPETokenizer:
    def __init__(self, vocab_size=vs):
        """
        初始化多语种BPE分词器
        
        参数:
            vocab_size: 词汇表大小
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>"]
        
    def train(self, file_patterns):
        """
        训练分词器
        
        参数:
            file_patterns: 文件路径模式列表，例如:
                ["data/english.txt", "data/chinese.txt", "data/spanish.txt", 
                 "data/french.txt", "data/german.txt", "data/japanese.txt"]
        """
        # 1. 初始化BPE模型
        self.tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
        
        # 2. 设置规范化器 - 使用NFKC标准化Unicode字符
        self.tokenizer.normalizer = Sequence([NFKC()])
        
        # 3. 设置预分词器 - 字节级别处理，适合多语言
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # 4. 设置解码器
        self.tokenizer.decoder = decoders.ByteLevel()
        
        # 5. 设置后处理器 - 添加特殊标记
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # 6. 定义训练器
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        # 7. 收集训练文件
        training_files = []
        for pattern in file_patterns:
            if isinstance(pattern, list):
                training_files.extend(pattern)
            else:
                training_files.append(pattern)
        
        print(f"使用 {len(training_files)} 个文件训练分词器...")
        
        # 8. 训练分词器
        self.tokenizer.train(files=training_files, trainer=trainer)
        
        print(f"分词器训练完成，词汇表大小: {self.tokenizer.get_vocab_size()}")
        
    def save(self, directory):
        """保存分词器到指定目录"""
        os.makedirs(directory, exist_ok=True)
        self.tokenizer.save(f"{directory}/tokenizer.json")
        
        # 保存配置
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        with open(f"{directory}/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print(f"分词器已保存到 {directory}/")
    
    def load(self, directory):
        """从指定目录加载分词器"""
        self.tokenizer = Tokenizer.from_file(f"{directory}/tokenizer.json")
        
        # 加载配置
        with open(f"{directory}/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            self.vocab_size = config.get("vocab_size", vs)
            self.special_tokens = config.get("special_tokens", ["<|endoftext|>", "<|unk|>", "<|pad|>"])
            
        print(f"分词器已加载，词汇表大小: {self.tokenizer.get_vocab_size()}")
    
    def encoding(self, text):
        """
        编码函数：将字符串转换为整数ID列表
        
        参数:
            text: 输入字符串
            
        返回:
            IDs列表
        """
        if self.tokenizer is None:
            raise ValueError("分词器未初始化，请先训练或加载分词器")
            
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decoding(self, ids):
        """
        解码函数：将整数ID列表转换回字符串
        
        参数:
            ids: ID列表
            
        返回:
            解码后的字符串
        """
        if self.tokenizer is None:
            raise ValueError("分词器未初始化，请先训练或加载分词器")
            
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self):
        """获取词汇表大小"""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

# 使用示例
if __name__ == "__main__":
    # 1. 初始化分词器
    tokenizer = MultilingualBPETokenizer(vocab_size=vs)
    
    # 2. 定义训练文件路径
    # 假设你的六个语言文件在同一目录下，命名为:
    # language_files = [
    #     "data/english.txt",
    #     "data/chinese.txt", 
    #     "data/spanish.txt",
    #     "data/french.txt",
    #     "data/german.txt",
    #     "data/japanese.txt"
    # ]
    language_files = ["./data/input.txt"]
    
    
    if os.path.exists("./tokenizer/tokenizer.json"):
        print("检测到已存在的分词器，直接加载...")
        tokenizer.load("./tokenizer")
    else:
        tokenizer.train(language_files)
        tokenizer.save("./tokenizer")
    
    # 5. 测试编码和解码函数
    test_texts = [
        """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:"""
    ]
    
    print("\n测试编码和解码:")
    for text in test_texts:
        # 编码
        ids = tokenizer.encoding(text)
        print(f"\n原文: {text}")
        print(f"编码: {ids}")
        
        # 解码
        decoded = tokenizer.decoding(ids)
        print(f"解码: {decoded}")
        print(f"匹配: {text == decoded}")
    
    # 6. 演示如何加载已保存的分词器
    print("\n" + "="*50)
    print("演示加载已保存的分词器:")
    
    # new_tokenizer = MultilingualBPETokenizer()
    # new_tokenizer.load("my_multilingual_tokenizer")
    
    # # 测试加载的分词器
    # test_text = "This is a test with multiple languages: 中文, Español, Français"
    # ids = new_tokenizer.encoding(test_text)
    # decoded = new_tokenizer.decoding(ids)
    
    # print(f"原文: {test_text}")
    # print(f"编码: {ids}")
    # print(f"解码: {decoded}")
    # print(f"匹配: {test_text == decoded}")