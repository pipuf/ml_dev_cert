from conftest import HOME_URL #,PATIENT_LANDING_URL,PROVIDER_LANDING_URL
from pages.home_page import HomePage

PATIENT_LANDING_URL = "Patient/Index"
PROVIDER_LANDING_URL = "Doctor/Visits"
SITEADMIN_LANDING_URL = "Admin/PersonSearch"
CALLCENTER_LANDING_URL = ""
SCHEDULER_LANDING_URL = ""
NURSE_LANDING_URL = ""

def loginUI(home):
    title = home.title
    assert title == "Teladoc"
    
def test_valid_login_patient(home):
    home.login("matthew.williams","Pass@word1")
    assert home.wait_for_url(HOME_URL+PATIENT_LANDING_URL)
    # assert home.driver.current_url == HOME_URL+PATIENT_LANDING_URL

def test_valid_login_provider(home):
    home.login("cpprovider","Pass@word1")
    assert home.wait_for_url(HOME_URL+PROVIDER_LANDING_URL)

def test_valid_login_admin(home):
    home.login("cpsiteadminqa","Pass@word1")
    assert home.wait_for_url(HOME_URL+SITEADMIN_LANDING_URL)

def test_wrong_username(home):
    home.login("sarasa","Pass@word1")
    assert home.read_error_message("Invalid login attempt.") 

def test_wrong_password(home):
    home.login("matthew.williams","Pass")
    assert home.read_error_message("Invalid login attempt.")