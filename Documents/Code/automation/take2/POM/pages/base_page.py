from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as exCon 

class BasePage(object):

    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(self.driver,30)

    def wait_for_element(self, locator):
        return self.wait.until(exCon.presence_of_element_located(locator))
    
    def wait_for_text(self, locator, text):
        return self.wait.until(exCon.text_to_be_present_in_element(locator, text))

    def find(self,locator):
        return self.driver.find_element(*locator)
    
    def wait_for_url(self, url):
        return self.wait.until(exCon.url_to_be(url))

