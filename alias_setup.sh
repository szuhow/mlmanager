#!/bin/bash

SHELL_RC="$HOME/.bashrc"  # or "$HOME/.zshrc" for Zsh

echo "alias docker-compose='docker compose'" >> "$SHELL_RC"
echo "alias python='python3'" >> "$SHELL_RC"

echo "Aliases added to $SHELL_RC. Please restart your terminal or run:"
echo "source $SHELL_RC"
