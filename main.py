import requests
import json

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
import os
import re
from target import target


def sanitize_title(title):
    """
    通过替换文件名中不允许的字符，对给定的标题进行清理。

    参数:
        title (str): 需要清理的标题。

    返回:
        str: 清理后的标题。
    """
    return re.sub(r'[\\/*?:"<>|]', "", title)


def get_reference(title):
    """
    从 Crossref API 中检索给定标题的参考信息。

    参数:
        title (str): 文章的标题。

    返回:
        str: 包含文章信息的参考字符串。
            如果给定标题没有找到任何项目，则返回 "未找到项目"。
            如果在请求过程中发生错误，则返回 "发生错误"。
    """
    # 将空格替换为 '+'
    title = title.replace(' ', '+')

    # 向 Crossref API 发送请求
    response = requests.get(
        f"https://api.crossref.org/works?query.title={title}&select=title,author,created,container-title,volume,issue,page,published-print,published-online,type,ISSN,publisher")

    # 检查请求是否成功
    if response.status_code == 200:
        data = json.loads(response.text)

        # 检查是否找到任何项目
        if data['message']['items']:
            # 获取第一个项目
            item = data['message']['items'][0]

            # 准备参考字符串
            ref_string = ""

            # 添加记录类型
            ref_string += f"%0 {item['type']}\n" if 'type' in item else ""

            # 添加标题
            ref_string += f"%T {item['title'][0]}\n"

            # 添加作者
            for author in item['author']:
                if 'given' in author and 'family' in author:
                    ref_string += f"%A {author['family']}, {author['given']} \n"
                elif 'given' in author:
                    ref_string += f"%A {author['given']}\n"
                elif 'family' in author:
                    ref_string += f"%A {author['family']}\n"

            # 添加期刊名称
            ref_string += f"%J {item['container-title'][0]}\n"

            # 添加卷号
            ref_string += f"%V {item['volume']}\n" if 'volume' in item else ""

            # 添加期号
            ref_string += f"%N {item['issue']}\n" if 'issue' in item else ""

            # 添加页码
            ref_string += f"%P {item['page']}\n" if 'page' in item else ""

            # 添加 ISSN
            if 'ISSN' in item and len(item['ISSN']) > 0:
                ref_string += f"%@ {item['ISSN'][0]}\n"

            # 添加出版年份
            if 'published-print' in item:
                ref_string += f"%D {item['published-print']['date-parts'][0][0]}\n"
            elif 'published-online' in item:
                ref_string += f"%D {item['published-online']['date-parts'][0][0]}\n"
            else:
                ref_string += f"%D {item['created']['date-parts'][0][0]}\n"

            # 添加出版商
            ref_string += f"%I {item['publisher']}\n" if 'publisher' in item else ""

            return ref_string
        else:
            return "未找到项目。"
    else:
        return "发生错误。"



def process(articles):
    """
    处理文章列表并检索参考文献信息。

    Args:
        articles (list): 包含文章内容的列表。

    Returns:
        None
    """
    # 定义提示的模板
    template = "现在给你一段引用的参考文献，我需要你找出其中的标题，并将标题返回给我，不需要加上引号或者其他字符，只需要标题。参考文献如下：{article}"

    # 创建 PromptTemplate 对象
    prompt = PromptTemplate(template=template, input_variables=["article"])

    # 创建 OpenAI 语言模型
    llm = OpenAI(temperature=0, model='gpt-3.5-turbo-instruct')

    # 创建 LLMChain 对象
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # 如果 'references' 目录不存在，则创建它
    if not os.path.exists('references'):
        os.makedirs('references')

    # 处理每篇文章并检索参考信息
    for i, article in tqdm(enumerate(articles, start=1)):
        # 使用语言模型生成标题
        title = llm_chain.run(article)
        title = title.strip('\n')
        
        try:
            # 获取标题的参考信息
            reference = get_reference(title)
            
            # 为参考文献创建文件路径
            path = os.path.join("references", '{}_{}.enw'.format(i, sanitize_title(title)))
            
            # 将参考文献写入文件
            with open(path, 'w') as f:
                f.write(reference)
        except Exception as e:
            print("第{}篇文章获取信息失败".format(i))
            print("失败原因为：", e)


if __name__ == '__main__':
    # 从目标文件中获取文章, 这需要你自己创建一个 target.py 文件，里面包含一个 target 变量，这个变量是一个字符串，里面包含了你的文章，并且每篇文章之间用 [ ] 分隔开
    articles = target.strip().split('[ ]')
    articles = articles[1:]

    # 打印文章数量
    print(len(articles))
    process(articles)
