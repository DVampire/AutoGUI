import pyautogui

def main():
    pyautogui.hotkey("ctrl", "win", "right")

    screenWidth, screenHeight = pyautogui.size()
    print(screenWidth, screenHeight)

    currentMouseX, currentMouseY = pyautogui.position()
    print(currentMouseX, currentMouseY)

    img = pyautogui.screenshot()
    img.save('my_screenshot.png')

if __name__ == '__main__':
    main()