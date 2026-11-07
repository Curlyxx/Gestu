# Gesture Control

Controla ventanas con **gestos de una mano en movimiento**, vistos por la cámara del equipo.
Funciona en Linux y Windows. Un proceso en segundo plano (el *daemon*) mantiene la cámara abierta y
ejecuta las acciones; un panel Tkinter sirve para grabar, entrenar, probar y encender/apagar el daemon,
como la app de un *live wallpaper*: la cierras y el daemon sigue corriendo.

## Cómo funciona

```
cámara ─► MediaPipe (21 puntos de la mano) ─► ventana deslizante de ~1.5 s
        ─► normalización ─► red Conv1D+GRU (TensorFlow) ─► gesto + confianza
        ─► filtros (umbral, votos, cooldown) ─► acción (Alt+Tab, Alt+F4, …)
```

No se entrena sobre píxeles sino sobre **landmarks**: es más ligero, no depende del fondo ni de la luz
y necesita pocos datos. Los videos se usan igual (se convierten a landmarks).

| Gesto | Acción por defecto |
|---|---|
| `swipe_derecha` | Siguiente ventana (Alt+Tab; los swipes seguidos encadenan sin soltar Alt) |
| `swipe_izquierda` | Ventana anterior (Alt+Shift+Tab) |
| `swipe_arriba` | Vista general / escritorio |
| `swipe_abajo` | Minimizar ventana activa |
| `agarrar` (palma abierta → puño) | Cerrar ventana (Alt+F4) — exige más confianza y un voto extra |
| `ninguno` | No hace nada (movimiento normal de la mano; **es la clase que evita falsos positivos**) |

## Estructura

```
gesture-control/
├── app.py                    # abre el panel
├── requirements.txt
├── config/
│   ├── settings.yaml         # cámara, umbrales, duración de clips, épocas…
│   └── actions.yaml          # gesto -> acción, y comando por sistema operativo
├── data/
│   ├── raw_videos/<gesto>/   # videos opcionales (.mp4 .avi .mov .mkv .webm)
│   ├── landmarks/<gesto>/    # clips .npz (los crea record.py / import_videos.py)
│   └── splits/splits.json    # qué clip va a train / val / test
├── models/                   # modelo .keras, labels, curvas, matriz de confusión
├── runtime/                  # pid, estado, log del daemon
├── scripts/
│   ├── record.py             # graba clips con la cámara
│   ├── import_videos.py      # videos -> landmarks
│   ├── build_dataset.py      # divide train/val/test
│   ├── train.py              # entrena
│   ├── test.py               # evalúa en test
│   ├── pipeline.py           # importar -> dataset -> entrenar -> testear
│   └── run.py                # el daemon
└── src/gesture_control/      # paths, features, hands, actions, model, gui, daemon…
```

Todas las rutas se calculan desde la raíz del proyecto (`pathlib`), nada es absoluto.

## 1. Crear carpetas y archivos (solo si no usas el zip)

La forma más corta: ejecuta `python crear_proyecto.py`, que crea todo el árbol y escribe todos los archivos.
A mano, solo el esqueleto:

**Linux / macOS**
```bash
mkdir -p gesture-control/{config,data/{raw_videos,landmarks,splits},models,runtime,scripts,src/gesture_control}
cd gesture-control
touch app.py requirements.txt .gitignore config/{settings,actions}.yaml
touch scripts/{record,import_videos,build_dataset,train,test,pipeline,run}.py
touch src/gesture_control/{__init__,paths,config,osinfo,features,camera,hands,actions,model,dataset,training,evaluation,status,daemon,gui}.py
```

**Windows (PowerShell)**
```powershell
$d = "data\raw_videos","data\landmarks","data\splits","models","runtime","scripts","config","src\gesture_control"
New-Item -ItemType Directory -Force -Path (@("gesture-control") + ($d | % { "gesture-control\$_" })) | Out-Null
cd gesture-control
"app.py","requirements.txt",".gitignore","config\settings.yaml","config\actions.yaml" | % { New-Item -ItemType File -Force $_ | Out-Null }
"record","import_videos","build_dataset","train","test","pipeline","run" | % { New-Item -ItemType File -Force "scripts\$_.py" | Out-Null }
"__init__","paths","config","osinfo","features","camera","hands","actions","model","dataset","training","evaluation","status","daemon","gui" | % { New-Item -ItemType File -Force "src\gesture_control\$_.py" | Out-Null }
```

## 2. Entorno virtual e instalación

Python 3.10–3.13. Si TensorFlow no publica rueda para tu versión de Python, usa 3.12 (con `uv`, `pyenv` o el
paquete de tu distro).

Tkinter viene aparte en Linux: `sudo pacman -S tk` (Arch) · `sudo apt install python3-tk` (Debian/Ubuntu) ·
`sudo dnf install python3-tkinter` (Fedora).

**Linux / macOS**
```bash
cd gesture-control
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

**Windows (PowerShell)**
```powershell
cd gesture-control
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1     # si lo bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

La primera vez que se usa la mano, se descarga `models/hand_landmarker.task` (de Google). Si no hay
internet, descárgalo manualmente de
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
y guárdalo con ese nombre en `models/`.

> MediaPipe eliminó `mediapipe.solutions` en versiones recientes; este proyecto usa solo la API *Tasks*
> (`HandLandmarker`), que funciona en las versiones nuevas.

## 3. Uso paso a paso

Abre el panel con `python app.py`. Arriba aparece **LISTO** (verde) o **FALTA: …** (rojo), con el detalle de
qué falta: clips por gesto, videos sin importar, división train/val/test, modelo y test. Los botones siguen
el orden del flujo:

1. **Grabar datos** (`scripts/record.py`). Se abre la cámara con el esqueleto de la mano dibujado.
   `1`–`6` elige el gesto, `ESPACIO` graba un clip (cuenta 3-2-1 y 1.5 s), `U` deshace el último,
   `V` guarda también el video `.mp4`, `Q` sale.
2. **Importar videos** (opcional). Pon videos en `data/raw_videos/<gesto>/`. Un video corto cuenta como
   un clip; uno largo se corta en trozos de 1.5 s (ideal para `ninguno`).
3. **Construir dataset** (`build_dataset.py`). Divide en train/val/test **por sesión de grabación** para
   que el test mida generalización real y no memoria.
4. **Entrenar** (`train.py`). Aumenta datos (ruido, escala, rotación leve, recortes de tiempo), usa
   *early stopping* y guarda el modelo.
5. **Testear** (`test.py`). Accuracy, reporte por clase y matriz de confusión (pestaña *Gráficas*).
   **Pipeline 2→5** encadena los pasos 2 a 5.

Todo también funciona por terminal: `python scripts/record.py`, `python scripts/pipeline.py --skip-import`, etc.

### Cuántos datos

- Mínimo 30 clips por gesto (configurable), mejor 60–100. Para `ninguno`, el doble.
- Graba en **al menos 3 sesiones separadas** (cierra y vuelve a abrir el grabador): cada apertura es una
  sesión. Con menos de 3 sesiones por gesto, el split usa bloques cronológicos, que protege menos.
- Varía distancia a la cámara, velocidad, iluminación y fondo. En `ninguno` incluye la mano quieta,
  mover el mouse, teclear, gesticular al hablar, rascarse… lo que harás mientras el daemon corre.
- Si cambias `camera.mirror` después de grabar, los swipes izquierda/derecha quedan invertidos: regraba.

## 4. El daemon (modo *live wallpaper*)

En el panel: **▶ Iniciar daemon** (necesita un modelo entrenado). Puedes cerrar la ventana; el daemon sigue
en segundo plano con la cámara abierta. Vuelve a abrir `python app.py` para ver su estado o detenerlo.

- **Dry-run**: detecta y muestra la acción sin ejecutarla. Se puede conmutar con el daemon en marcha.
  Recomendado para probar. El último gesto y la acción aparecen en el panel y en `runtime/daemon.log`.
- **Ventana de cámara**: muestra la cámara con el esqueleto (útil para depurar; `Q` la cierra).
- Línea de comandos: `python scripts/run.py [--dry-run] [--preview] [--background]`,
  `python scripts/run.py --status`, `python scripts/run.py --stop`.
- Mientras el daemon usa la cámara no se puede grabar; el panel te pide detenerlo antes.
- **Iniciar con la sesión**: Linux, añade `python /ruta/scripts/run.py --background` (con el python del
  `.venv`) a *Autostart*. Windows, un `.bat` con ese comando en `shell:startup`.

### Seguridad contra falsos positivos

Umbral de confianza (0.85; 0.95 para cerrar), 2 predicciones iguales seguidas (3 para cerrar), *cooldown* de
1 s tras cada acción y mínimo de frames con mano en la ventana. Todo en `config/settings.yaml`.

## 5. Linux: X11, Wayland, KDE

`osinfo` detecta el sistema y la sesión, y el panel muestra el backend elegido.

- **X11**: `pynput` funciona. Para minimizar instala `xdotool`.
- **Wayland** (por defecto en Plasma 6): `pynput` **no puede inyectar teclas**. Opciones:
  1. Iniciar sesión en X11.
  2. `ydotool`: instálalo, ten `ydotoold` corriendo y permiso sobre `/dev/uinput`
     (en Arch: `sudo pacman -S ydotool`, y activa su servicio; revisa la doc de tu distro).
     Con `backend: auto` se elige solo si está en el PATH.
  3. Para minimizar en KDE Wayland se usa `kdotool` (AUR). Si tu versión nombra distinto el comando,
     edítalo en `config/actions.yaml`.
- La vista general usa `meta+w` (KDE). En GNOME, cámbialo a `meta` en `actions.yaml`.

## 6. Windows

`pynput` funciona directo. Vista general = `win+tab`, minimizar = `win+down` (en una ventana maximizada la
primera vez solo la restaura), cerrar = `alt+f4`.

## 7. Personalizar

- **Acciones**: pestaña *Acciones* del panel (edita `config/actions.yaml`). Tipos: `keys: "ctrl+shift+m"`,
  `alt_tab: forward|backward`, `cmd: "comando externo"`. Cada acción puede definirse por `windows`, `linux`,
  `linux_wayland` o `all`.
- **Ajustes**: pestaña *Configuración* (cambia solo los valores, conserva los comentarios del YAML).
  Cambiar la duración del clip exige regrabar y reentrenar.
- **Gestos nuevos**: añade el nombre en `gestures:` de `settings.yaml` y en `gestures:` de `actions.yaml`,
  graba clips y reentrena. El modelo se reentrena desde cero.

## 8. Problemas comunes

| Síntoma | Qué revisar |
|---|---|
| No abre la cámara | `camera.index` en `settings.yaml`; cierra otras apps que la usen |
| No se descarga el modelo de manos | Ver el enlace manual en la sección 2 |
| Dispara sin querer | Más clips de `ninguno`, subir `threshold` y `votes` |
| No dispara | Revisa en dry-run con la ventana de cámara que la mano se detecta; baja `threshold`; más datos |
| Alt+Tab no hace nada en Linux | Casi seguro Wayland sin `ydotool` (el panel muestra una advertencia) |
| La accuracy de test es mucho menor que la de validación | Pocas sesiones de grabación: graba en más sesiones y condiciones |
