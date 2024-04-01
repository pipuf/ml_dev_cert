import pytest
from selenium import webdriver
from pages.home_page import HomePage

HOME_URL = "https://qa.statdoctorsapp.com/"

""" PATIENT_LANDING_URL = "Patient/Index"
PROVIDER_LANDING_URL = "Doctor/Visits"
SITEADMIN_LANDING_URL = "Admin/PersonSearch"
CALLCENTER_LANDING_URL = ""
SCHEDULER_LANDING_URL = ""
NURSE_LANDING_URL = "" """


@pytest.fixture
def driver():
    print ("driver started...")
    driver = webdriver.Firefox()
    driver.get(HOME_URL)
    yield driver
    # execute test case
    driver.quit()

@pytest.fixture
def home(driver):
    home = HomePage(driver)
    home.driver.get(HOME_URL)
    print ("Home page loaded...")
    return home
