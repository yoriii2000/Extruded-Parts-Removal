import pyautogui
import time

def click_on_chat():
    # 根據你的畫面上的具體位置調整座標
    x, y = 12, 687
    pyautogui.click(x, y)

def type_and_send_message():
    message = "=work"
    pyautogui.typewrite(message)
    pyautogui.press('enter')

def main():
    while True:
        click_on_chat()
        type_and_send_message()
        time.sleep(60)  # 等待兩分鐘

if __name__ == "__main__":
    main()
