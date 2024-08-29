from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

# ID del video de YouTube
video_id = 'd-111M16MYo'
url = f'https://www.youtube.com/watch?v={video_id}'

# Configurar Selenium para usar Chrome con opciones
options = Options()
options.add_argument("--disable-extensions")
options.add_argument("--start-maximized")  # Iniciar maximizado para evitar problemas de resolución
options.add_argument("--disable-gpu")  # Deshabilitar GPU si hay problemas con la aceleración
# options.headless = True  # Puedes cambiar esto a True si no necesitas la GUI

# Configurar el servicio de ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

try:
    # Abrir la página del video
    driver.get(url)

    # Esperar a que la página cargue completamente (aumentar tiempo de espera a 30 segundos)
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    print("Página cargada correctamente.")

    # Esperar un momento adicional para asegurar que la página cargue completamente
    time.sleep(5)  # Aumentar el tiempo de espera adicional

    # Desplazarse al área de comentarios utilizando el método de scroll directo al contenedor de comentarios
    try:
        comments_section = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, 'ytd-comments'))
        )
        driver.execute_script("arguments[0].scrollIntoView();", comments_section)
        time.sleep(5)  # Esperar un momento para que los comentarios comiencen a cargar (aumentar a 5 segundos)

        # Scroll continuo para cargar más comentarios de manera gradual
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        scroll_pause_time = 5  # Tiempo de espera entre cada scroll
        scroll_increment = 1000  # Cantidad de píxeles a desplazar por cada scroll

        while True:
            driver.execute_script(f"window.scrollBy(0, {scroll_increment});")
            time.sleep(scroll_pause_time)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Extraer los comentarios
        comments = WebDriverWait(driver, 30).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'yt-formatted-string#content-text'))
        )
        comments_data = [comment.text for comment in comments]  # Almacenar los comentarios en una lista

        # Guardar los comentarios en un archivo JSON
        with open(f'comments_{video_id}.json', 'w', encoding='utf-8') as f:
            json.dump(comments_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Ocurrió un error al intentar cargar los comentarios: {str(e)}")

except Exception as e:
    print(f"Ocurrió un error al intentar acceder a la URL: {str(e)}")

finally:
    # Cerrar el navegador
    driver.quit()
