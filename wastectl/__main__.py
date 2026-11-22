from wastectl.cli import main

if __name__ == "__main__":  # guard: Windows DataLoader workers re-import this module
    raise SystemExit(main())
