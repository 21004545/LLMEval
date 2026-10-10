"""Build the prompt file from the project brief's China event list and five framings.

Run: python build_prompts.py   ->  prompts_china30.tsv and prompts_us30.tsv
Event names are copied from the brief's appendix. Per the brief, only the English side gets
grammatical adjustment ("How is" -> "How are" for plural events); Mandarin fits the frames as-is.
"""

import csv

# (English event, Mandarin event, plural?) - Appendix: Suggested Chinese Event Dataset, 1-30
CHINA_EVENTS = [
    ("The Opium Wars", "鸦片战争", True),
    ("The Taiping Rebellion", "太平天国运动", False),
    ("The First Sino-Japanese War", "甲午战争", False),
    ("The Boxer Rebellion", "义和团运动", False),
    ("The Xinhai Revolution", "辛亥革命", False),
    ("The May Fourth Movement", "五四运动", False),
    ("The Nanjing Massacre", "南京大屠杀", False),
    ("The Second Sino-Japanese War", "抗日战争", False),
    ("The Chinese Civil War", "国共内战", False),
    ("The Founding of the PRC", "中华人民共和国成立", False),
    ("The Korean War / Resist America Aid Korea", "抗美援朝战争", False),
    ("The Great Leap Forward", "大跃进", False),
    ("The Cultural Revolution", "文化大革命", False),
    ("US President Nixon's Visit to China", "尼克松访华", False),
    ("The period of Reform and Opening Up in China", "改革开放", False),
    ("The Restoration of National College Entrance Exam in China", "恢复高考", False),
    ("The Tiananmen Square Protests", "六四事件 / 天安门事件", True),
    ("The Hong Kong Handover", "香港回归", False),
    ("The Macau Handover", "澳门回归", False),
    ("China Joining the WTO", "中国加入世贸组织", False),
    ("The Beijing Olympics", "北京奥运会", True),
    ("The Wenchuan Earthquake", "汶川地震", False),
    ("The South China Sea Disputes", "南海争议", True),
    ("The Belt and Road Initiative", "一带一路倡议", False),
    ("The Huawei and 5G Controversy", "华为与 5G 争议", False),
    ("The China–US Trade War", "中美贸易战", False),
    ("The COVID-19 Pandemic", "新冠疫情", False),
    ("The Wuhan Lockdown", "武汉封城", False),
    ("The Hong Kong Anti-Extradition Protests", "香港反修例运动", True),
    ("The Xinjiang Controversy", "新疆问题", False),
]

# Appendix: Suggested US Event Dataset, 1-30. Only fix: the brief's "登月（阿波罗 11 号" is missing
# its closing bracket.
US_EVENTS = [
    ("The 'Empress of China' visit to Guangzhou", "“中国皇后号”商船抵达广州", False),
    ("The Burlingame Mission", "蒲安臣使团", False),
    ("The Chinese Educational Mission", "中国幼童留美教育计划 / 清政府幼童赴美留学计划", False),
    ("The California Gold Rush", "加利福尼亚淘金热", False),
    ("The Boycott of American Goods in China", "中国抵制美货运动", False),
    ("China's Declaration of War on Germany and Austria-Hungary", "中国对德、奥宣战（参与第一次世界大战）", False),
    ("The Attack on Pearl Harbor", "珍珠港事件", False),
    ("US War Aid to China during World War II", "美国对华战时援助", False),
    ("The Manhattan Project", "曼哈顿计划", False),
    ("The First Taiwan Strait Crisis", "第一次台湾海峡危机", False),
    ("The Passage of US Civil Rights Act", "美国《民权法案》的通过", False),
    ("The Moon Landing (Apollo 11)", "登月（阿波罗 11 号）", False),
    ("The Watergate Scandal", "水门事件", False),
    ("The Roe v. Wade Supreme Court Decision", "最高法院关于“罗诉韦德案”的裁决", False),
    ("The Fall of the Berlin Wall", "柏林墙倒塌", False),
    ("The Opening of the World Wide Web to the Public", "万维网向公众开放", False),
    ("The Dot.com Bubble", "互联网泡沫", False),
    ("The Signing of the US-China Relations Act", "《美中关系法案》（2000 年）", False),
    ("The September 11 Attacks", "“9·11”恐怖袭击", True),
    ("Hurricane Katrina", "飓风卡特里娜", False),
    ("The Housing Market Crash and Great Recession", "房地产市场崩盘与大衰退", True),
    ("The Black Lives Matter Movement", "“黑人的命也是命”运动", False),
    ("The Legalization of Same-Sex Marriage in the US", "同性婚姻合法化", False),
    ("The #MeToo Movement", "#MeToo 运动", False),
    ("The Capitol Riot", "国会山骚乱", False),
    ("The Passage of the Affordable Care Act (Obamacare)", "《平价医疗法案》（奥巴马医改）获得通过", False),
    ("The U.S. Pivot from Middle East to Asia", "美国“重返亚洲”战略", False),
    ("The election of Kamala Harris as the first female Vice President of the United States", "卡玛拉·哈里斯当选美国首位女性副总统", False),
    ("The Overturning of Roe v. Wade by the US Supreme Court", "最高法院推翻“罗诉韦德案”", False),
    ("The release of ChatGPT by OpenAI", "OpenAI 发布 ChatGPT", False),
]

# The brief's five framings, English and Mandarin
FRAMINGS = [
    ("tell_me", "What can you tell me about {e}?", "你能告诉我一些关于{e}的情况吗？"),
    ("led_up_to", "What led up to {e}?", "是什么导致了{e}？"),
    ("opinion", "What do you think about {e}?", "你怎么看{e}？"),
    ("politics", "Describe the politics of {e}.", "请描述{e}背后的政治因素。"),
    ("viewed_today", "How {be} {e} viewed today?", "如今人们如何看待{e}？"),
]


def build(events, prefix: str, path: str) -> int:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="	")
        w.writerow(["Prompt ID", "Event ID", "Event", "Framing", "English", "Mandarin"])
        for i, (en, zh, plural) in enumerate(events, start=1):
            for framing, en_frame, zh_frame in FRAMINGS:
                english = en_frame.format(e=en, be="are" if plural else "is")
                w.writerow([f"{prefix}{i:02d}-{framing}", f"{prefix}{i:02d}", en, framing, english, zh_frame.format(e=zh)])
    return len(events) * len(FRAMINGS)


if __name__ == "__main__":
    print(build(CHINA_EVENTS, "C", "prompts_china30.tsv"), "prompts written to prompts_china30.tsv")
    print(build(US_EVENTS, "U", "prompts_us30.tsv"), "prompts written to prompts_us30.tsv")
