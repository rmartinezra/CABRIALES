# Publicar este proyecto en GitHub

Esta carpeta esta lista para manejarse con Git localmente. El repositorio no
debe incluir corridas completas ni datos pesados: `run_*`, `.shw`, `.tar.gz` y
`.hgt` quedan ignorados por `.gitignore`.

## 1. Inicializar y crear el primer commit

```bash
git init
git add .
git commit -m "Initial Machin muography pipeline"
```

## 2. Crear el repositorio remoto

Opcion A: desde la pagina de GitHub

1. Crear un repo nuevo en GitHub, por ejemplo `machin-muography-pipeline`.
2. No marcar "Add README", porque este proyecto ya tiene README.
3. Copiar la URL SSH o HTTPS.

Opcion B: con GitHub CLI, si esta instalado y autenticado

```bash
gh repo create machin-muography-pipeline --private --source . --remote origin --push
```

## 3. Enlazar este repo local al remoto

Con SSH:

```bash
git remote add origin git@github.com:TU_USUARIO/machin-muography-pipeline.git
git branch -M main
git push -u origin main
```

Con HTTPS:

```bash
git remote add origin https://github.com/TU_USUARIO/machin-muography-pipeline.git
git branch -M main
git push -u origin main
```

## 4. Antes de subir

Revisar que no entren archivos grandes:

```bash
git status --short
git check-ignore -v data/machin10dia.tar.gz run_machin_10dia
```

Si `git status` muestra `run_*`, `.shw`, `.tar.gz` o `.hgt`, no hagas push
todavia: hay que corregir el `.gitignore` o sacar esos archivos del index.
