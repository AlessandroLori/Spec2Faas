from dataclasses import dataclass
from typing import Any

@dataclass
class TestCodeMessage:
    specification: str
    function_signature: str
    prompt: bool

@dataclass
class TestCodeResult:
    content: str
    time: float
    tokens: float
    ctx: any

@dataclass
class TestDeployMessage:
    code:str

@dataclass
class TestDeployResult:
    result: str
    time: float
    tokens: float
    ctx: any
    deployed_function: str = ""
    invocation_attempts: int = 0

@dataclass
class TestExecCodeMessage:
    specification: str
    function_signature: str
    code: str
    tests: str
    system: bool = False

@dataclass
class TestExecCodeResult:
    final_function: str
    passed: bool
    time: float
    tokens: float
    attempts: int
    tokens_debugger:float = 0
    time_debugger: float = 0
    first_error: str = ""
    last_error: str = ""

@dataclass
class TestDebugMessage:
    specification: str
    code: str
    error_message: str
    new_chat: bool


@dataclass
class TestDebugResult:
    code: str
    tokens: float

@dataclass
class TestMessage:
    content:str

@dataclass
class TestMessageResult:
    content: str
    time : float
    tokens: float
    type: str = ""


@dataclass
class TestSystemMessage:
    tokens: dict
    time: dict
    # Number of messages exchanged
    messages: int = 1
    # From Assistant
    # The generated prompt
    prompt: str = ""
    # From Entry Point
    # The generated signature
    signature: str = ""
    # From Coder
    # The string of code of the original function
    original_func: str = ""
    # The code of the original function
    code: str = ""
    new_chat: bool = False
    # From test designer
    # The code of the tests
    tests: str = ""
    # The string of code of the tests
    tests_str: str = ""
    sender: str = ""
    # From debugger and test executor
    first_error: str = ""
    last_error: str = ""
    # The string of code of the final function
    final_func: str = ""
    # The code of the final function
    code_final_func: str = ""
    # The number of attempts
    attempts: int = 0
    # If the function is generated
    generated: bool = False
    # From deployer
    # The name of the handler
    result_deployment: str = ""
    # If the function is deployed
    deployed: bool = False
    # Only used in EntryPoint
    type: str = ""
    # The string of code of the deployed function (function +
    deployed_function: str = ""