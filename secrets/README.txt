Encrypted instance credentials

comet.env.gpg contains COMET_API_KEY for the authorized Comet account.
The decryption passphrase was sent to the user in the chat on 2026-10-29; it is not stored in Git.

Restore on a new instance (GnuPG required):
  umask 077
  gpg --output /tmp/disciples2-comet.env --decrypt secrets/comet.env.gpg
  cat /tmp/disciples2-comet.env >> /workspace/.env
  rm /tmp/disciples2-comet.env
  chmod 600 /workspace/.env
  set -a
  . /workspace/.env
  set +a

GPG asks for the passphrase interactively. Do not put it in shell arguments or source files.
