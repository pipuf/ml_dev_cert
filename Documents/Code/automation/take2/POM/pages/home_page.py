from pages.base_page import BasePage
from selenium.webdriver.common.by import By

class HomePage(BasePage):
    USERNAME_INPUT = (By.NAME,"username")
    PASSWORD_INPUT = (By.NAME, "password")
    LOGIN_CTA = (By.CSS_SELECTOR, "button[type=submit]")
    ERROR_LABEL =(By.CSS_SELECTOR,".portal-error-line > span")
    CREATE_ACCOUNT_CTA = ()
    SSO_CTA = ()
    
    def login(self,username,password):
        self.wait_for_element(self.USERNAME_INPUT).send_keys(username)
        self.wait_for_element(self.PASSWORD_INPUT).send_keys(password)
        self.find(self.LOGIN_CTA).click()

    def read_error_message(self, text):
        return self.wait_for_text(self.ERROR_LABEL,text)

    def nav_to_reset_password(self):
        pass

    def nav_to_create_account(self):
        pass

    def SSO_MyChart(self):
        pass