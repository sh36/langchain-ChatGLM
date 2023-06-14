# import _thread as thread
# import base64
# import datetime
# import hashlib
# import hmac
# import json
# from urllib.parse import urlparse
# import ssl
# from datetime import datetime
# from time import mktime
# from urllib.parse import urlencode
# from wsgiref.handlers import format_date_time

# import websocket

# from abc import ABC
# import requests
# from typing import Optional, List
# from langchain.llms.base import LLM

# from models.loader import LoaderCheckPoint
# from models.base import (RemoteRpcModel,
#                          AnswerResult)
# from typing import (
#     Collection,
#     Dict
# )




# class Ws_Param(object):
#     # 初始化
#     def __init__(self, APPID, APIKey, APISecret, gpt_url):
#         self.APPID = APPID
#         self.APIKey = APIKey
#         self.APISecret = APISecret
#         self.host = urlparse(gpt_url).netloc
#         self.path = urlparse(gpt_url).path
#         self.gpt_url = gpt_url

#     # 生成url
#     def create_url(self):
#         # 生成RFC1123格式的时间戳
#         now = datetime.now()
#         date = format_date_time(mktime(now.timetuple()))

#         # 拼接字符串
#         signature_origin = "host: " + self.host + "\n"
#         signature_origin += "date: " + date + "\n"
#         signature_origin += "GET " + self.path + " HTTP/1.1"

#         # 进行hmac-sha256进行加密
#         signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
#                                  digestmod=hashlib.sha256).digest()

#         signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

#         authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

#         authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

#         # 将请求的鉴权参数组合为字典
#         v = {
#             "authorization": authorization,
#             "date": date,
#             "host": self.host
#         }
#         # 拼接鉴权参数，生成url
#         url = self.gpt_url + '?' + urlencode(v)
#         # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
#         return url


# # 收到websocket错误的处理
# def on_error(ws, error):
#     print("### error:", error)


# # 收到websocket关闭的处理
# def on_close(ws):
#     print("### closed ###")


# # 收到websocket连接建立的处理
# def on_open(ws):
#     thread.start_new_thread(run, (ws,))


# def run(ws, *args):
#     data = json.dumps(gen_params(appid=ws.appid, question=ws.question))
#     ws.send(data)


# # 收到websocket消息的处理
# def on_message(ws, message):
#     data = json.loads(message)
#     code = data['header']['code']
#     if code != 0:
#         print(f'请求错误: {code}, {data}')
#         ws.close()
#     else:
#         choices = data["payload"]["choices"]
#         status = choices["status"]
#         content = choices["text"][0]["content"]
#         print(content, end='')
#         if status == 2:
#             ws.close()


# def gen_params(appid, question):
#     """
#     通过appid和用户的提问来生成请参数
#     """
#     data = {
#         "header": {
#             "app_id": appid,
#             "uid": "1234"
#         },
#         "parameter": {
#             "chat": {
#                 "domain": "general",
#                 "random_threshold": 0.5,
#                 "max_tokens": 2048,
#                 "auditing": "default"
#             }
#         },
#         "payload": {
#             "message": {
#                 "text": [
#                     {"role": "user", "content": question}
#                 ]
#             }
#         }
#     }
#     return data


# def main(appid, api_key, api_secret, gpt_url, question):
#     wsParam = Ws_Param(appid, api_key, api_secret, gpt_url)
#     websocket.enableTrace(False)
#     wsUrl = wsParam.create_url()
#     ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
#     ws.appid = appid
#     ws.question = question
#     ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})


# if __name__ == "__main__":
#     # 测试时候在此处正确填写相关信息即可运行
#     main(appid="89ed2ea4",
#          api_key="0736d4e16d9cd2bb55d4b1bd38d07229",
#          api_secret="MDlkZTRlYzNlZTcwZmU5YjU3ZmVmNWNj",
#          gpt_url="wss://spark-api.xf-yun.com/v1.1/chat",
#          question="你是谁？你能做什么？")
    



# def _build_message_template() -> Dict[str, str]:
#     """
#     :return: 结构
#     """
#     return {
#         "role": "",
#         "content": "",
#     }


# class XINGHUOLLM(RemoteRpcModel, LLM, ABC):
    
#     wsParam = Ws_Param.Ws_Param("89ed2ea4", "0736d4e16d9cd2bb55d4b1bd38d07229", "MDlkZTRlYzNlZTcwZmU5YjU3ZmVmNWNj", "wss://spark-api.xf-yun.com/v1.1/chat")
#     websocket.enableTrace(False)
#     wsUrl = wsParam.create_url()
#     api_base_url: str = wsUrl
#     model_name: str = "xinghuo"
#     max_token: int = 10000
#     temperature: float = 0.01
#     top_p = 0.9
#     checkPoint: LoaderCheckPoint = None
#     history = []
#     history_len: int = 10

#     def __init__(self, checkPoint: LoaderCheckPoint = None):
#         super().__init__()
#         self.checkPoint = checkPoint

#     @property
#     def _llm_type(self) -> str:
#         return "FastChat"

#     @property
#     def _check_point(self) -> LoaderCheckPoint:
#         return self.checkPoint

#     @property
#     def _history_len(self) -> int:
#         return self.history_len

#     def set_history_len(self, history_len: int = 10) -> None:
#         self.history_len = history_len

#     @property
#     def _api_key(self) -> str:
#         pass

#     @property
#     def _api_base_url(self) -> str:
#         return self.api_base_url

#     def set_api_key(self, api_key: str):
#         pass

#     def set_api_base_url(self, api_base_url: str):
#         self.api_base_url = api_base_url

#     def call_model_name(self, model_name):
#         self.model_name = model_name

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         pass

#     # 将历史对话数组转换为文本格式
#     def build_message_list(self, query) -> Collection[Dict[str, str]]:
#         build_message_list: Collection[Dict[str, str]] = []
#         history = self.history[-self.history_len:] if self.history_len > 0 else []
#         for i, (old_query, response) in enumerate(history):
#             user_build_message = _build_message_template()
#             user_build_message['role'] = 'user'
#             user_build_message['content'] = old_query
#             system_build_message = _build_message_template()
#             system_build_message['role'] = 'system'
#             system_build_message['content'] = response
#             build_message_list.append(user_build_message)
#             build_message_list.append(system_build_message)

#         user_build_message = _build_message_template()
#         user_build_message['role'] = 'user'
#         user_build_message['content'] = query
#         build_message_list.append(user_build_message)
#         return build_message_list

#     def generatorAnswer(self, prompt: str,
#                         history: List[List[str]] = [],
#                         streaming: bool = False):

#         try:
#             import openai
#             # Not support yet
#             openai.api_key = "EMPTY"
#             openai.api_base = self.api_base_url
#         except ImportError:
#             raise ValueError(
#                 "Could not import openai python package. "
#                 "Please install it with `pip install openai`."
#             )
#         # create a chat completion
#         completion = openai.ChatCompletion.create(
#             model=self.model_name,
#             messages=self.build_message_list(prompt)
#         )

#         history += [[prompt, completion.choices[0].message.content]]
#         answer_result = AnswerResult()
#         answer_result.history = history
#         answer_result.llm_output = {"answer": completion.choices[0].message.content}

#         yield answer_result


import json
import ssl
import websocket

from abc import ABC
from typing import Optional, List, Dict, Collection

from models.loader import LoaderCheckPoint
from models.base import RemoteRpcModel, AnswerResult
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
from urllib.parse import urlencode, urlparse
import base64
import hmac
import hashlib
from urllib.parse import urlparse
from wsgiref.handlers import format_date_time


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, gpt_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候使用的url，出现问题时可以复制此url进行测试
        print('### url:', url)
        return url


class XINGHUOLLM(RemoteRpcModel, ABC):
    #wsParam = Ws_Param("Your APPID", "Your APIKey", "Your APISecret", "wss://api.xf-yun.com/v1/chat")
    wsParam = Ws_Param("89ed2ea4", "0736d4e16d9cd2bb55d4b1bd38d07229", "MDlkZTRlYzNlZTcwZmU5YjU3ZmVmNWNj", "wss://spark-api.xf-yun.com/v1.1/chat")
    def __init__(self, checkPoint: Optional[LoaderCheckPoint] = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "XINGHUO"

    @property
    def _check_point(self) -> Optional[LoaderCheckPoint]:
        return self.checkPoint

    def generatorAnswer(self, prompt: str, history: Optional[List[List[str]]] = None, streaming: bool = False):
        websocket.enableTrace(False)
        wsUrl = self.wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=self.on_message, on_error=self.on_error, on_close=self.on_close)
        ws.on_open = self.on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def on_open(self, ws):
        self.send_message(ws, "init")

    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            print(content, end='')
            if status == 2:
                ws.close()

    def on_error(self, ws, error):
        print("### error:", error)

    def on_close(self, ws):
        print("### closed ###")

    def gen_params(self, appid, question):
        data = {
            "header": {
                "app_id": appid,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    "domain": "general",
                    "random_threshold": 0.5,
                    "max_tokens": 2048,
                    "auditing": "default"
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "user", "content": question}
                    ]
                }
            }
        }
        return data
    #建立历史信息list，存储历史信息
    def build_message_list(self, query) -> Collection[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "System message"
            },
            {
                "role": "user",
                "content": query
            }
        ]
        return messages
        
# 测试代码能力的函数
def test_code_ability():
    # 创建XINGHUOLLM对象
    xinghuollm = XINGHUOLLM()
    
    # 设置测试输入
    prompt = "Hello, how are you?"
    
    # 执行测试
    xinghuollm.generatorAnswer(prompt)


# 执行测试代码能力的函数
test_code_ability()