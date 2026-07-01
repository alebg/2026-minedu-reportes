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

El análisis tiene dos fases. En ambas se excluyen las secciones de bibliografía
para que los porcentajes reflejen reuso de texto sustantivo, no citas académicas
compartidas.

**Fase 1: Escaneo de 8-gramas.** Se extraen ventanas deslizantes de 8 palabras
consecutivas de ambos documentos (normalizados a minúsculas, sin tildes ni
puntuación). Cada secuencia de 8 palabras idénticas cuenta como coincidencia
textual. Esta técnica (document fingerprinting mediante n-gramas) pertenece a la
familia de métodos de string-matching usados en detección académica de plagio,
junto con algoritmos como Rabin-Karp y Knuth-Morris-Pratt empleados por
Turnitin y CopyCatch (Hamed et al., *Frontiers in Computer Science*, 2025,
https://doi.org/10.3389/fcomp.2025.1504725).

Se reportan dos métricas:
- *Texto de la consultora en la publicación*: porcentaje de las palabras del
  trabajo original que aparecen textualmente en la publicación oficial.
  Responde a: ¿cuánto del trabajo original fue copiado?
- *Texto de la publicación copiado de la consultora*: porcentaje de las
  palabras de la publicación que provienen del trabajo de la consultora.
  Responde a: ¿cuánto de la publicación es texto copiado?

**Fase 2: Alineación por párrafos.** Cada párrafo del documento fuente se
compara con todos los párrafos de la publicación usando el índice de Jaccard:
la cantidad de 8-gramas compartidos dividida por la cantidad total de 8-gramas
distintos entre ambos párrafos
(https://es.wikipedia.org/wiki/%C3%8Dndice_de_Jaccard).

Se eligió el índice de Jaccard por tres razones: (1) está normalizado entre 0 y
1, lo que permite comparar párrafos de distinta longitud; (2) es simétrico, sin
privilegiar una dirección sobre la otra; (3) penaliza el relleno, ya que agregar
texto alrededor de una copia hace crecer la unión sin aumentar la intersección.

100% = texto idéntico. Sobre 50% = copiado sustancial con ediciones menores.
Se excluyen coincidencias por debajo de 15%.

**Verificación manual.** Los pasajes mostrados provienen de los archivos .txt
extraídos de los documentos originales. Para verificar un pasaje en el PDF o
DOCX original usando Ctrl+F, busque fragmentos cortos (5-10 palabras), no el
pasaje completo. Los saltos de línea y de página pueden impedir que Ctrl+F
encuentre frases largas.

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

El resumen se imprime por salida estándar. Los reportes completos (todos los
pasajes y alineaciones) se escriben en `output/`, un archivo `.txt` por cada
par de documentos comparados.

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
