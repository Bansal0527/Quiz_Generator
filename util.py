import PyPDF2
import json
import traceback

# def parse_file(file):
#     if file.name.endswith(".pdf"):
#         try:
#             pdf_reader= PyPDF2.PdfReader(file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#             return text
#         except PyPDF2.util.PdfReadError:
#             raise Exception("Error reading the PDF file.")
#
#
#     elif file.name.endswith(".txt"):
#         return file.read().decode("utf-8")
#
#     else:
#         raise Exception(
#             "Unsupported file format"
#         )



RESPONSE_JSON = {
    "1": {
        "no": "1",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer"
    },
    "2": {
        "no": "1",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer"
    }
}

def get_table_data(quiz_str):
    try:
        # convert the quiz from a str to dict
        quiz_dict = json.loads(quiz_str)
        quiz_table_data = []
        # Iterate over the quiz dictionary and extract the required information
        for key, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " | ".join(
                [
                    f"{option}: {option_value}"
                    for option, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
        return quiz_table_data
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
