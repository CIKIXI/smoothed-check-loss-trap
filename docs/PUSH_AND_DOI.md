# Publish this repository (2 steps, then fill in the DOI)

This directory is already a **local Git repository with a `v1.0.0` tag**, and the remote is
configured as

```
origin  git@github.com:CIKIXI/smoothed-check-loss-trap.git      (SSH)
```

SSH authentication with your GitHub account already works on this machine
(`ssh -T git@github.com` answers `Hi CIKIXI!`), so pushing needs no password or token.
Everything that can be prepared offline has been prepared; what remains is creating the empty
repository on GitHub and minting the DOI.

---

## Step 1: create the empty GitHub repository and push

1. Open <https://github.com/new>.
2. Owner: **CIKIXI**. Repository name: **smoothed-check-loss-trap**. Visibility: **Public**.
3. Do **not** tick "Add a README file", ".gitignore" or "license": the repository must be
   empty, otherwise the push is rejected.
4. Create the repository, then in PowerShell:

   ```powershell
   cd C:\Users\liangxin2\Desktop\SCI1\code_release
   git push -u origin main
   git push origin v1.0.0
   ```

5. Check on GitHub: 99 files, and the tag `v1.0.0` listed under **Releases / Tags**.

If the repository name or account differs, point the remote at the real one first:

```powershell
git remote set-url origin git@github.com:<account>/<repo>.git
```

> **Commit identity.** The commit is authored as `Kaixu Cai
> <CIKIXI@users.noreply.github.com>`, the GitHub no-reply address of the `CIKIXI` account, so
> GitHub links the commit to your profile.

## Step 2: archive on Zenodo and mint the DOI

1. Sign in at <https://zenodo.org> with the GitHub account (button "Sign in with GitHub").
2. Go to **Settings → GitHub** (or <https://zenodo.org/account/settings/github/>) and press
   **Sync now** if the new repository is not listed yet.
3. Flip the switch for **smoothed-check-loss-trap** to ON. Zenodo now watches it for releases.
4. On GitHub, open the repository and create a release from the tag:
   **Releases → Draft a new release → Choose a tag: v1.0.0 → Release title** (e.g. "Code and
   results for the TEST submission") **→ Publish release**.
5. Refresh the Zenodo GitHub page: a new entry appears with a DOI. Two DOIs are shown:
   - **concept DOI** — "Cite all versions"; always resolves to the newest version.
     **This is the one to put in the paper.**
   - **version DOI** — pins `v1.0.0`.
6. Open the Zenodo record and check the metadata (title, author Cai Kaixu, licence MIT,
   keywords). You can edit it; the files come from the release automatically.

## Step 3: put the real DOI into the manuscripts

```powershell
cd C:\Users\liangxin2\Desktop\SCI1
python simulations/set_zenodo_doi.py 10.5281/zenodo.1234567
```

The script writes the DOI into `paper/main.tex` and `simulations/build_test.py`, rebuilds both
submission packages, and reports whether any placeholder is left. To drop the DOI clause and
keep only the repository link, run it with `none` instead of a DOI.

## Step 4: sanity check before submitting

- [ ] Both URLs resolve in a private/incognito window: the GitHub repository and the Zenodo
      record (they must be genuinely public).
- [ ] The Zenodo record contains the scripts **and** the saved `simulations/*.json` results.
- [ ] The repository shows 99 files and the tag `v1.0.0`.
- [ ] `XXXXXXX` no longer appears in `paper/main.tex` or `simulations/build_test.py`.
- [ ] `submission_TEST/main_test.pdf` was rebuilt after the DOI change and shows the DOI in
      the Declarations.
- [ ] The DOI quoted in the paper is the **concept** DOI, not the version DOI.
