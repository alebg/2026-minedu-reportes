# Análisis de plagio: productos de consultoría vs publicaciones MINEDU

Herramienta para comparar entregables de consultoría contra publicaciones
oficiales del Ministerio de Educación del Perú (MINEDU) y detectar reuso
textual sin atribución.

## Contexto

Tres estudios publicados por MINEDU en marzo 2026 contienen texto proveniente
de productos de consultoría desarrollados por GRADE para el programa PMESUT.
Las publicaciones oficiales no mencionan a las consultoras responsables del
trabajo original. Este análisis cuantifica el grado de coincidencia textual
entre los documentos fuente y las publicaciones.

## Pares de documentos comparados

| Producto (consultoría)               | Publicación (MINEDU)                                                                                          |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Producto 3 (v170524)                  | Estudio para la identificación de desigualdades y barreras de acceso y permanencia en la ESU con enfoque de género |
| Producto 5 (Informe 08012025)         | Propuestas para la reducción de desigualdades en el Acceso, Permanencia y Egreso                              |
| Producto 2 (v06.03.2024)              | Sistematización de experiencias y buenas prácticas para la igualdad de género en el ámbito universitario      |

## Metodología

El análisis tiene dos fases:

**Fase 1: Escaneo de 8-gramas.** Se extraen ventanas deslizantes de 8 palabras
consecutivas de ambos documentos (normalizados a minúsculas, sin tildes ni
puntuación). Cada secuencia de 8 palabras idénticas cuenta como coincidencia
textual. Esta técnica (document fingerprinting mediante n-gramas) pertenece a la familia
de métodos de string-matching usados en detección académica de plagio, junto con
algoritmos como Rabin-Karp y KMP empleados por Turnitin y CopyCatch (Hamed et
al., *Frontiers in Computer Science*, 2025,
https://doi.org/10.3389/fcomp.2025.1504725).

**Fase 2: Alineación por párrafos.** Cada párrafo del documento fuente se
compara con todos los párrafos de la publicación usando el índice de Jaccard
(https://es.wikipedia.org/wiki/%C3%8Dndice_de_Jaccard). Se reportan métricas
agregadas y los párrafos con mayor similitud, lado a lado.

## Requisitos

- Python 3.12+
- Dependencias: `python-docx`, `pymupdf`, `attrs`
- Dev: `mypy`, `pytest`, `ruff`

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install python-docx pymupdf attrs

python3 compare.py
```

El reporte se imprime por salida estándar. Para guardarlo:

```bash
python3 compare.py > reporte.txt 2>&1
```

## Tests

```bash
pip install mypy pytest ruff
mypy compare.py
pytest tests/ -v
```

## Responsable del análisis

**Luis Bordo**
COO y cofundador, Dream Aim Deliver AI (Suiza).
Candidato a doctor en Filosofía de la Ciencia, Université de Genève.
Sociólogo licenciado, Pontificia Universidad Católica del Perú (PUCP).

Investigador social cuantitativo con 8 años de experiencia en diseño de
encuestas, modelamiento estadístico y análisis de datos, incluyendo trabajo
en el Ministerio de Educación del Perú (MINEDU) y la PUCP. Formación doctoral
en causalidad e inferencia causal aplicada a las ciencias sociales. Competencias
técnicas: Python, Rust, R, SQL, modelamiento estadístico, ingeniería de datos.

LinkedIn: https://www.linkedin.com/in/luisbordo/
ORCID: https://orcid.org/0009-0008-3172-1413

## Licencia

MIT (ver LICENSE).
