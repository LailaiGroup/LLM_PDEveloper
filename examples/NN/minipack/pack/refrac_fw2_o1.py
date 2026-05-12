# langchain libs
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph, START
from langchain_openai.chat_models.base import BaseChatOpenAI
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
from pydantic import BaseModel, Field

# OS libs
import getpass
import os

OA_key = os.environ["OPENAI_API_KEY"]
DPSK_key = os.environ["DPSK_API_KEY"]

from typing import List, TypedDict
from langgraph.graph import END, StateGraph, START


# Data model
class code_cache(BaseModel):
    """Schema for code solutions to questions about XLB."""

    files: List[str] = Field(description="The paths of files to be modified.")
    codes: List[str] = Field(description="List of generated codes to be added to corresponding file. Length of this list is as long as files.")
    # test_code: str = Field(description="the file to test the newly generated codes, ")

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        paper: the paper we want to reproduce
        src_code: the original XLB codes
        modified: the code added new codes for very now
        stage: mark the current stage ["first generation, correction, merged"]
        first generation : Code solution for first time
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        
        iterations : Number of tries
    """
    paper: str
    shorter_src_code: str
    original_src_code: code_cache # the very original source codes
    stage: str

    coding_notes: str # some notes while coding, stored at the .history/

    # generation record
    first_generation: code_cache
    generation_messages: List[str]
    generation_error: str
    generation_iterations: int

    # test flag
    test_code_cache:code_cache
    test_pass:bool
    test_error:str

    # correction record
    correction: code_cache
    correction_messages:List
    correction_error: str
    correction_iterations: int

# Functions
def merge(ori_code:code_cache, add_code:code_cache)->(bool, code_cache):
    ori_code
    modified_code_cache = code_cache(files=ori_code.files, codes=ori_code.codes)
    mod_files_list = modified_code_cache.files
    mod_files_codes = modified_code_cache.codes

    if ("test.py" not in files):
        return False, modified_code_cache, "no test.py files in it"
    if len(add_code.files) != len(add_code.codes):
        print("codes not match the files")
        return False, modified_code_cache, "number of codes doesn't match the number of files"
    else:
        gen_wrapper = "\n\n#<BEG> LLM added <\BEG>\n\n{generation}\n\n#<END> LLM added <\END>\n\n"
        for i in range(len(add_code.files)):
            if (add_code.files[i] not in ['src/base.py', 'src/boundary_conditions.py', 'src/lattice.py', 'src/models.py', 'src/utils.py', 'test.py']):
                print("{} file path doesn't exist".format(add_code.files[i]))
                return False, modified_code_cache , "{} file path doesn't exist".format(add_code.files[i])
            file = add_code.files[i]
            file_ind = mod_files_list.index(file)
            mod_files_codes[file_ind] = mod_files_codes[file_ind] + gen_wrapper.format(generation = add_code.codes[add_code.files.index(file)])

        return True, modified_code_cache, "successfully merged"

## check the math and the format
def math_check():
    return 

## from code_cache to str, transform the whole src/ into the .md strings
def generate_mannual(src_code: code_cache)->str:
    with open("./.backup/mod_code/mod_src_code.md") as f:
        mod_mannual = f.read()
    for i in range(len(src_code.files)):
        section = "\n## {0} \n ```python\n{1}\n```\n".format(src_code.files[i], src_code.codes[i])
        mod_mannual = mod_mannual + section
    return mod_mannual

## transform the code_cache to executable envrionment
def code_cache2env(mod_code_cache: code_cache):
    files = mod_code_cache.files
    codes = mod_code_cache.codes
    for i in range(len(files)):
        f = open(files[i],'w')
        f.write(codes[i])
        f.close()

## refresh the environment
def refresh():
    import os 
    os.system("rm -r ./src/*")
    os.system("rm test.py")
    os.system("touch src/.placeholder")

def backup_media(backup_path, i):
    os.mkdir(backup_path+str(i))
    bpath = backup_path+str(i)
    os.system("cp -r ./src/ {}".format(bpath))
    # os.system("cp -r ./mod_code {}".format(bpath))
    os.system("cp -r ./test.py {}".format(bpath))

## transition functions
def transition(state: GraphState):
    if not state["test_pass"]:
        return "corrector"
    else:
        return "end"

# Functions /end



## models 
### format llm
code_fmt_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Here is a code base called XLB:\n{ori_code_mannual} \n\
You are a coding assistant specialized in python.\n\
You would be given a suggestion about code modification generated by other llm.\n\
Your tasks:
1. The suggestion you recieved is not formal, your task is to transform it into formal format. The formal format is a special class consist of two lists.\n\
2. All the modifications in given suggestion are additional, which means the new codes would be directly appended to the ends of files. Your output should be two lists: one includes the name of the files to be added, and the other one should include the codes to be added.\n\
For example, if "test0.py" would be added "print("hello world!")" and "src/utils.py" added "print("byebye")", your output would be like ["test0.py","src/utils.py"], ["print("hello world!")","print("byebye")"]\n\ 
Here are some notes:\n\
1. You should always provide the full path of the names of files. e.g. src/base.py but not only base.py;\n\
2. There would be a special file called "test.py" for testing. You should never miss it in the given suggestion. And the content of "test.py" should not be empty.\n\
3. Do not output unnecessary string, like "\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n". It will make our framework fail.\n\
Here is the generation from other LLM:\n\
{generation}""",
        ),
        ("placeholder", "{fmt_messages}"),
    ]
)
fmt_llm_id = "gpt-4o"
print("fmt llm id: "+fmt_llm_id)
llm_fmt = ChatOpenAI(temperature=0.1, model=fmt_llm_id)
code_fmt_chain_oai = code_fmt_prompt | llm_fmt.with_structured_output(code_cache)

# llm_id = "claude-3-5-sonnet-20240620"
# llm_fmt = ChatAnthropic(temperature=0.05, model=llm_id, max_tokens=4096)
# code_fmt_chain_oai = code_fmt_prompt | llm_fmt.with_structured_output(code_cache)

### expensive o1-preview
# code_llm = ChatOpenAI(temperature=1, model=code_llm_id) # o1-preview not supported the max_tokens

code_llm_id = 'o1-2024-12-17'
code_llm = ChatOpenAI(model=code_llm_id) # o1-preview not supported the max_tokens

print("code llm card: "+code_llm_id)
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
    ]
)
code_gen_chain_oai = code_gen_prompt | code_llm 
code_sys_prompt_template = '''
Role:\n\
You are an expert in computational physicist, partial differential equations (PDEs), scientific computing, and lattice Boltzmann method (LBM).\n\

You are a specialized agent within a multi-agent collaborative system for code generation. Your specific role is to generate implementation code for the PDE solver and/or its associated boundary conditions (BCs) in the given paper.

The code snippets are for a PDE solver within the XLB library-a LBM framework built with Python and JAX. (a Python library developed by Google)
\n\ 
Materials:\n\
The XLB library is:\n\
{src_code}\n\
The specific numerical algorithm to implement for the PDE solver and/or its associated BCs is described in the paper:
\n{paper}\n\
\n\ 
\n\
Steps of task:\n\
Your task consists of three steps: \n\
    Step 1. Generate your code for the solver or boundary conditions, as described in the paper under "Materials", to the user. Ensure that your submitted aligns with the XLB's structure and style.\n\
    Step 2. Based on the test case in the paper, generate a Python file named "test.py". This file is used to verify the functionality of your implementation. It should produce both `.vtk` and `.png` files for visualization, following the output methods used in Python file "test_template.py"\n\
    Step 3. Submit your code to the user in the format specified in the "Requirements of Submission" section below. \n\
\n\
Requirements of submission:\n\
    1. Your submission including the generated code should clearly list each file that requires modification. For each file, provide the **complete** code to be appended at the end.\n\
       **Example:**

   - **File to Modify:** `src/models.py`

   - **Action:** Append the following code to the end of `src/models.py`:

     ```python
     class ExampleSolver:
         # Complete implementation of the ExampleSolver class
     ```\n\
    2. The code you provide will be directly appended to the specified files. Design your code to integrate seamlessly with the library XLB, maintaining consistency in style and structure.\n\
    3. Your submission MUST include a Python file called "test.py". The user validates your code by executing "test.py". Refer to the existing `test_template.py` for guidance. Ensure that `test.py` is capable of generating both `.vtk` and `.png` files during execution.\n\ 

To provide effective corrections, you must understand the process of how other agents in your multi-agent system apply them. This process consists of three steps.

Step 1. All the code segments from other LLMs between ”#<BEG> LLM added <\BEG>“ and “#<END> LLM added <\END>” will be removed.\n\ 

Step 2. Another agent receives your correction.\n\

Step 3. Another agent appends 
the corrected code from your correction 
to 
the end of the files the correction specifies.
\n\
Noted:\n\
Some IMPORTANT POINTS you need to keep in mind:\n\
    1. Do not alter the structure of the original XLB codebase.\n\
    2. Ensure that your code in submission is complete and adheres to the style of the original XLB library.\n\
    3. Your submission must include a test file named "test.py", which the user will execute to verify the solver or/and BCs code.\n\
    4. The "test.py" file should produce both '.vtk' and '.png' files during testing, following the same visualization output method as outlined in "test_template.py".\n\
    5. According to the established workflow process, provide incremental updates and clearly identify which file(s) are being modified.\n\
    6. Do not create any new files.\n\
\n\
Here are some notes for coding based on the pass errors:\n\
{coding_notes}
'''

### corrector llm
# corr_llm_id = "o1-preview"
# corr_llm = ChatOpenAI(temperature=1, model=corr_llm_id) # o1-preview not supported the max_tokens


corr_llm_id = 'o1-2024-12-17'
corr_llm = ChatOpenAI(model=corr_llm_id) # o1-preview not supported the max_tokens
print("corr llm: "+corr_llm_id)

corr_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
    ]
)
corr_gen_chain_oai = corr_gen_prompt | corr_llm 
corrector_sys_prompt_template = '''
Role:\n\
You are an expert in computational physicist, partial differential equations (PDEs), scientific computing, and lattice Boltzmann method (LBM).\n\


You are a specialized agent within a multi-agent collaborative system for code generation. Your specific role is to analyze and correct the code snippets assigned to you.
The code snippets are for a PDE solver within the XLB library-a LBM framework built with Python and JAX. (a Python library developed by Google)
\n\ 

Materials:\n\
The specific numerical algorithm to implement for the PDE solver and/or its associated boundary conditions (BCs) is described in the paper:
\n\n {paper} \n\n\

You will receive a composite codebase comprising the original XLB part and another part contributed by an external LLM, where the latter is enclosed by the markers “#<BEG> LLM added <\BEG>” and “#<END> LLM added <\END>”. This composite codebase is: \n\ {modified_src_code} \n\

The compilation errors encountered by the Python compiler while executing the test file "test.py" within the composite codebase are:\n\ {err_msg} \n\

Steps of task:\n\

You need to correct the code snippets of the PDE solver to the user. 

Your task consists of three steps: \n\
    Step 1. Generate a complete code implementation of the numerical algorithm described in the given paper by adapting the external LLM's code and resolving the encountered compilation errors. \n\ 
    Step 2. Based on the test case mentioned in the given paper, generate "test.py" for the user to validate your code. \n\
    Step 3. Provide a brief summary of your correction, formatted according to the guidelines detailed in the "Requirements of correction" section below. \n\
    
Requirements of correction:\n\
    You should clearly list each file that needs to be corrected. 
    
    For each file, output
    the **complete** code block that will be appended to the end of that file after your correction. Your output should strictly adhere to the format demonstrated in the following example.
    \n\
       **Example of your output correction**

   - **File to be corrected:** `src/models.py`

   - **Action:** Append the following code to the end of `src/models.py`:

    - **Correct code:**
     ```python
     class ExampleSolver:
         # Complete implementation of the ExampleSolver class
     ```\n\
\n\

To provide effective corrections, you must understand the process of how other agents in your multi-agent system apply them. This process consists of three steps.

Step 1. All the code segments from other LLMs between ”#<BEG> LLM added <\BEG>“ and “#<END> LLM added <\END>” will be removed.\n\ 

Step 2. Another agent receives your correction.\n\

Step 3. Another agent appends 
the corrected code from your correction 
to 
the end of the files the correction specifies.
\n\


Noted:\n\
Some IMPORTANT POINTS you need to keep in mind:\n\
    1. Do not alter the structure of the original XLB code base.\n\
    2. Ensure that your code is **complete** and adheres to the style of the original XLB library.\n\
    3. Include a test file named "test.py", which the user will execute to verify the solver or boundary condition code.\n\
    4. The "test.py" file should produce both '.vtk' and '.png' files during testing, following the same visualization output method as outlined in "test_template.py".\n\
    5. According to the established workflow process, provide incremental updates and clearly identify which file(s) are being modified.\n\
    6. Do not create any new files beyond those specified.\n\
\n\
Here are some notes for coding based on the pass errors:\n\
{coding_notes} \n\
'''

### error correction summary agent
err_sum_llm_id = "gpt4-o"
err_sum_llm = ChatOpenAI(temperature=0.2, model=err_sum_llm_id) # o1-preview not supported the max_tokens
err_sum_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
    ]
)
err_sum_chain_oai = code_gen_prompt | err_sum_llm
err_sum_sys_prompt_template = '''
Setting:\n\
Role: You are now an expert in python and Jax (numerical library developed by google). Your responsibility is to maintain a history file to record the error user has made\n\
Matterials: Here is the XLB lib user are coding with:\n{ori_code_mannual}\n\
Tasks:\n\
You would be given a syntax error from user while running the code of Jax.\n\
You have 2 tasks: \n\
    1. Check whether the given errors have been in the history file;\n\
    2. If the errors have not been recorded, give a summary of the input error; \n\
\n\
Examples:\n\
input:\n\
Error:  /sscratch/hwu/miniconda3/envs/langgraph/lib/python3.11/site-packages/scipy/__init__.py:155: UserWarning: A NumPy version >=1.18.5 and <1.26.0 is required for this version of SciPy (detected version 1.26.4
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
Traceback (most recent call last):
  File "/sscratch/hwu/Desktop/autogen/Creativity_test/equilibrium_bc_creativity/solver/o1/lab/lab7/test.py", line 7, in <module>
    from src.models import *
  File "/sscratch/hwu/Desktop/autogen/Creativity_test/equilibrium_bc_creativity/solver/o1/lab/lab7/src/models.py", line 250, in <module>
    from jax import jit, partial
ImportError: cannot import name 'partial' from 'jax' (/sscratch/hwu/miniconda3/envs/langgraph/lib/python3.11/site-packages/jax/__init__.py)

summary of error:
NumPy Version Mismatch Warning: The installed version of NumPy is 1.26.4, which is incompatible with SciPy. SciPy requires NumPy versions between 1.18.5 and 1.26.0.\n\
ImportError in JAX: The script is trying to import partial from JAX, but it's not found in the version of JAX installed. This may indicate that partial is either missing in the version of JAX you're using or there is a conflict in your environment.\n\
\n\
\n\
Format of output:
No extra things but just a string. The format of string is "[type of error]: [the essential summary of the error]"
\n\

Notes:\n\
1. To save the space, your summary should be essential;\n\
2. For the syntax code specified for the XLB code (like the attributes and class not defined), you should also point out and record;\n\


'''
## models \end

# Agent nodes
def generate(state:GraphState):
    messages = state["generation_messages"]
    paper = state["paper"]
    shorter_src_code = state["shorter_src_code"]
    messages += [
            (
                "user",
                code_sys_prompt_template.format(paper=state["paper"], src_code = shorter_src_code, coding_notes=state["coding_notes"]),
            )
        ]
    raw_code =  code_gen_chain_oai.invoke({ 
                               "messages": [
                                   ("user",code_sys_prompt_template.format(paper=state["paper"], src_code = shorter_src_code, coding_notes=state["coding_notes"]))]
                                   })
    print("first gen pass")
    with open("st_gen.md","w") as f:
        f.write(raw_code.content)
    # format the results
    ori_code_mannual = generate_mannual(state["original_src_code"])
    with open("tmpmannual.txt","w") as f:
        f.write(ori_code_mannual)
    fmt_messages = [("user",f"Hello! please help me to extract the generated code and format them.")]
    fmt_code = code_fmt_chain_oai.invoke({"ori_code_mannual":ori_code_mannual, "generation":raw_code.content, "fmt_messages":fmt_messages})
    
    # try to merge the codes into the original libs
    m_state, ts_code_cache, m_descrip = merge(state["original_src_code"],fmt_code)
    while(not m_state):
        fmt_messages += [("user", "Sorry, seems that {}?".format(m_descrip))]
        fmt_code = code_fmt_chain_oai.invoke({"ori_code_mannual":ori_code_mannual, "generation":raw_code.content, "fmt_messages":fmt_messages})
        m_state, ts_code_cache, m_descrip = merge(state["original_src_code"],fmt_code)
    return {
        "first_generation": fmt_code,
        "stage": "first_generation",
        "test_code_cache": ts_code_cache
    }

def correct(state:GraphState):
    ts_err = state["test_error"]
    ts_code_cache = state["test_code_cache"]
    iters = state["correction_iterations"] + 1
    print(state["correction_iterations"])
    
    mod_code_mannual = generate_mannual(ts_code_cache)
    with open('corr_src_code{}.txt'.format(iters),'w') as f:
        f.write(mod_code_mannual)
    Ori_code_mannual = generate_mannual(state["original_src_code"])
    sys_prompt = corrector_sys_prompt_template.format(modified_src_code=mod_code_mannual, err_msg = ts_err, ori_code_mannual=Ori_code_mannual,paper=state["paper"],coding_notes=state["coding_notes"])
    req = sys_prompt

    print("---{ind} env refreshed ---".format(ind=state['correction_iterations']))
    r_code = corr_gen_chain_oai.invoke({"placeholder": "you are a good assistant.", 
                               "messages": [
                                   ("user",req)]
                                   }
                            )
    with open("correct_record.md","a") as f:
        f.write("\n iters "+str(state["correction_iterations"]) +" \n"+r_code.content)
    fmt_messages = [("user",f"hello! please help me to extract the generated code.")]
    ori_code_mannual = generate_mannual(state["original_src_code"])
    r_fmt = code_fmt_chain_oai.invoke({'ori_code_mannual':Ori_code_mannual,"generation":r_code.content, "fmt_messages":fmt_messages})
    merged_state, merged_result, m_descrip = merge(state["original_src_code"], r_fmt)
    
    corr_msg = [iters, ts_err, r_code]
    corr_msgs = state["correction_messages"]
    corr_msgs.append(corr_msg)
    return {
        "correction": r_fmt,
        "stage": "correction",
        "test_code_cache": merged_result,
        "correction_iterations": iters,
        "correction_messages": corr_msgs
    }
    
def test(state:GraphState):
    refresh()
    ts_code_cache = state["test_code_cache"]
    code_cache2env(ts_code_cache)
    import traceback
    import subprocess
    try:
        os.system("export PYTHONPATH=.")
        os.system("echo $PYTHONPATH")
        result = subprocess.run(['python3','test.py'],
                                capture_output = True,
                                text = True,
                                check = True)
        print("output: ", result.stdout)
        print("pass")
        return {"test_pass":True}
    except subprocess.CalledProcessError as e:
        print("Error: ", e.stderr)
        backup_media("./.testbackup/", state["correction_iterations"])
        # print(str(e.stderr))
        return {"test_pass":False, "test_error":str(e.stderr)}

# def test_code(test_file):
#     import traceback
#     import subprocess
#     f = open(test_file,'r')
#     test_c = f.read()
#     f.close()
    
#     try:
#         os.system("export PYTHONPATH=.")
#         result = subprocess.run(['python3','test.py'],
#                                 capture_output = True,
#                                 text = True,
#                                 check = True)
#         print("output: ", result.stdout)
#         return True, "Test past!"
#     except subprocess.CalledProcessError as e:
#         # print("Error: ", e.stderr)
#         return False, str(e.stderr)

# Agent nodes /end


workflow = StateGraph(GraphState)

workflow.add_node("generator", generate)  # generation solution
workflow.add_node("tester", test)
workflow.add_node("corrector", correct)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "tester")
workflow.add_edge("corrector", "tester")
# workflow.add_edge("tester",END)
workflow.add_conditional_edges(
    "tester",
    transition,
    {
        'end':END,
        'corrector':"corrector"
    }
)
app = workflow.compile()


# Main function

# refresh the environment
os.system("rm -r ./src/*")
os.system("rm -r ./test.py")
os.system("rm -r ./.testbackup/*")
print("environment refreshed")



# load the paper
import sys
with open(sys.argv[1]) as f:
    paper = f.read()

# load the original source codes
files = ["test.py","src/models.py", "src/test_template.py" ,"src/utils.py" ,"src/base.py","src/boundary_conditions.py","src/lattice.py"]
codes = []
for file in files:
    with open("./.backup/"+file) as f:
        codes.append(f.read())
original_src_code = code_cache(files=files, codes=codes)


# load essential shorter version of src codes
shorter_src_code = generate_mannual(original_src_code)
with open("shorter_src_code.md","w") as f:
    f.write(shorter_src_code)

# load the coding notes
with open(".err_history/coding_notes.md") as f:
    coding_notes = f.read()
with open("correct_record.md",'w') as f:
    f.write("correction records: \n")

solution = app.invoke({"original_src_code":original_src_code,
 "paper":paper, "shorter_src_code":shorter_src_code, "coding_notes":coding_notes,
 "stage":"START", "generation_messages": [], 
 "generation_iterations": 0, "correction_iterations": 0, "error":"",
 "correction_messages":[]})
a = solution["test_code_cache"]
# print("meta data:")
# print(solution.usage_metadata)
