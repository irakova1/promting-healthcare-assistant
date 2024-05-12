from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

MSG_END = "<|im_end|>"
_DEFAULT_MEDICAL_MESSAGES_TEMPLATE = [(
                        "system",
                        "You're an assistant who's good at medical support. Respond in a short manner",
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),]

DEFAULT_MEDICAL_TEMPLATE = ChatPromptTemplate.from_messages(_DEFAULT_MEDICAL_MESSAGES_TEMPLATE)

# FIXME: add requirment for context usage!
_OLD_MEDICAL_MESSAGES_TEMPLATE = [
    (
        "system",
        f"You're an assistant who's good at medical support. Respond in a short manner",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
]

OLD_MEDICAL_TEMPLATE = ChatPromptTemplate.from_messages(_OLD_MEDICAL_MESSAGES_TEMPLATE)

_DEFAULT_MEDICAL_MESSAGES_TEMPLATE = [
    (
        "system",
        f"You're an AI assistant who's good at medical support. Answer the questions based on the context below. Respond in a short manner. Human and AI end each message with {MSG_END}.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("ai", ""),
]

DEFAULT_MEDICAL_TEMPLATE = ChatPromptTemplate.from_messages(
    _DEFAULT_MEDICAL_MESSAGES_TEMPLATE
)

_DEFAULT_INSTRUCT_MEDICAL_MESSAGES_TEMPLATE = [
    (
        "system",
        "[INST] You are an AI medical assistant. Refer to and explicitly mention the context values in your response, tailor your advice to these specific measurements, "
        + 'and assess whether the indicators are within normal ranges. For general questions like "How can I improve my health?", assess each point of available context and provide tailored advice accordingly. '
        + "Offer detailed explanations for answer. Answer is limited up to 100 words.\n"
        + "Example:\nHuman: I've got my blood test. I have 8.0 mg/dL Calcium level, Vitamin D 19 ng/mL, Parathyroid Hormone level 37 pg/mL. Use this as context for next questions.\n"
        + "Human: Based on my blood test results (Calcium 8.0 mg/dL, Vitamin D 19 ng/mL, Parathyroid Hormone 37 pg/mL), what specific foods should I eat to improve my bone health? [/INST]\n"
        + "AI: 1. Increase your intake of fortified plant milks and juices, which can help boost your calcium levels, particularly important since your calcium is on the lower end of the normal range. "
        + "2. For Vitamin D, which is below optimal, consider fatty fish like salmon or mackerel, and also think about sensible sun exposure. 3. Foods rich in magnesium and vitamin K2 such as nuts and "
        + "fermented dairy can aid in the proper utilization of calcium and support bone health. Additionally, monitor your phosphorus intake as it directly affects calcium absorption.\n"
        # + "[INST] End of example.\nExample of required refferences:\nHuman: I am 27-years-old. I have a BMI of 25. My average step count is 3700 steps per day. I have an average total sleep time of 7.0 hours over the past week and an average of 3 waking hours"
        # + "per night. Use this as context for the next questions.\nHuman: How can I improve my overall health? [/INST]\n"
        # + "AI: For your age of 28 and your average step count around 3,700 steps per day ... . BMI of 25 ... . Average of 7 hours of sleep ... ."
        + "[INST] End of example. Previous examples are for illustration only and should not influence your answers. Forget all provided context and measurements.\n",
        # f"Human and AI end each message with {MSG_END}.",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input} [/INST]"),
    ("ai", ""),
]

DEFAULT_INSTRUCT_MEDICAL_TEMPLATE = ChatPromptTemplate.from_messages(
    _DEFAULT_INSTRUCT_MEDICAL_MESSAGES_TEMPLATE
)

_DEFAULT_MEDICAL_CONVERSATION_TEMPLATE = (
    f"""The following is a friendly conversation between a human and an AI assistant who's good at medical support. \
AI answers the questions based on the context below. The AI respond in a short manner. If the AI does not know the answer to a question, it truthfully says it does not know. Human and AI end each message with {MSG_END}."""
    + """Current conversation:
{history}
Human: {input}
AI: """
)
DEFAULT_MEDICAL_CONVERSATION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=_DEFAULT_MEDICAL_CONVERSATION_TEMPLATE,
)
