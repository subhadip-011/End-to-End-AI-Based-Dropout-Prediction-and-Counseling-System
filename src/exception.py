import sys
from src.logger import logger

def get_error_details(error, error_detail: sys):
    """
    Extracts file name, line number, and error message
    from the exception traceback.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error)

    return f"Error in [{file_name}] at line [{line_number}]: {error_message}"

class CustomException(Exception):
    """
    Custom exception that logs the error automatically
    whenever it is raised anywhere in the project.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message, error_detail)
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        raise CustomException(e, sys)