import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=chrome_options
)
URL = "https://jdvqemzzwpxzpmkxhribpl.streamlit.app/about"
try:
    print(f"Visiting URL: {URL }")
    driver.get(URL)
    wait = WebDriverWait(driver, 20)
    wake_up_button = wait.until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//button[contains(text(), 'back up') or contains(., 'back up')]",
            )
        )
    )
    print("App is asleep! Clicking the 'Wake Up' button...")
    wake_up_button.click()
    time.sleep(5)
    print("Wake up command sent successfully!")
except Exception as e:
    print("App is already awake or the wake-up button was not found.")
finally:
    driver.quit()