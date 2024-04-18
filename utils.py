from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper

def generate_script(subject, video_length, creativity, user_api_key):
    title_template = ChatPromptTemplate.from_messages(
        [
            ("human", """请为{subject}主题视频想一个吸引人的标题""")
        ]
    )
    script_template = ChatPromptTemplate.from_messages(
        [
            ("human",
            """你是一位短视频博主，根据以下标题和相关信息，创作一个视频脚本。
            视频标题：{title}，时长{duration}分钟，生成的脚本长度尽可能按照视频时长要求。视频要求开头抓住眼球，中间提供干货内容，结尾有惊喜。脚本格式按照【开头-中间-结尾】分隔。整体风格轻松有趣，吸引年轻人。
            视频内容可结合以下维基百科信息，但仅作为参考：'''{wikipedia_search}'''""")
        ]
    )

    model = ChatOpenAI(
        openai_api_key = user_api_key, temperature = creativity
    )

    title_chain = title_template | model
    script_chain = script_template | model

    title = title_chain.invoke({"subject": subject}).content

    search = WikipediaAPIWrapper(lang = "zh")
    search_result = search.run(subject)

    script = script_chain.invoke({"title":title,"duration": video_length, "wikipedia_search": search_result}).content
    return search_result, title, script

## print(generate_script("武装直升机",3,0.7,"sk-proj-OscPw6Gp2L0XMCr4ej5dT3BlbkFJ0hO1BOKNflGP8QckG7Rs"))

