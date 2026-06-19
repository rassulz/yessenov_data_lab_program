"""
emailer.py — the one tiny "agentic" action: email a chat summary to the administrator.

STRICT RULES (see CLAUDE.md / project brief):
  * Send ONLY to the student's own inbox (ALLOWED_ADMIN). Never to a user-typed address.
    The recipient is hard-pinned in code and cannot be overridden by config or the chat.
  * Send only on an explicit button click — never inside a loop. One click = one email.

Transport: prefers Gmail SMTP (set GMAIL_ADDRESS + GMAIL_APP_PASSWORD in secrets), and falls
back to MailerSend if a MailerSend key is configured. Secrets live in secrets.toml, never in code.
"""
from __future__ import annotations
from rag import load_secrets

# The ONLY address this app is ever allowed to email (the student's own inbox). Pinned in code
# so a stray env var or edited config can't redirect mail elsewhere.
ALLOWED_ADMIN = "rasulzeynullaer@gmail.com"


def _send_via_gmail(cfg: dict, subject: str, html_body: str, text_body: str, recipient: str) -> dict:
    import smtplib, ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    user = cfg.get("GMAIL_ADDRESS", "").strip()
    pw = cfg.get("GMAIL_APP_PASSWORD", "").replace(" ", "")     # app passwords are shown with spaces
    if not user or not pw:
        return {"ok": False, "error": "GMAIL_ADDRESS / GMAIL_APP_PASSWORD не заданы в secrets.toml"}

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Yessenov Data Lab — ассистент <{user}>"
    msg["To"] = recipient
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=30) as s:
            s.login(user, pw)
            s.sendmail(user, [recipient], msg.as_string())
        return {"ok": True, "message_id": "gmail-smtp", "to": recipient}
    except smtplib.SMTPAuthenticationError:
        return {"ok": False, "error": "Gmail отклонил вход — нужен App Password (не обычный пароль) "
                                      "и включённая 2FA. Проверьте GMAIL_APP_PASSWORD."}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _send_via_mailersend(cfg: dict, subject: str, html_body: str, text_body: str, recipient: str) -> dict:
    api_key = cfg.get("MAILERSEND_API_KEY", "")
    sender = cfg.get("SENDER_EMAIL", "info@app.commit.kz")
    if not api_key:
        return {"ok": False, "error": "Нет ни Gmail, ни MailerSend ключа в secrets.toml"}
    try:
        from mailersend import MailerSendClient, EmailBuilder
        ms = MailerSendClient(api_key=api_key)
        email = (EmailBuilder().from_email(sender, "Yessenov Data Lab — ассистент")
                 .to_many([{"email": recipient, "name": "Администратор"}])
                 .subject(subject).html(html_body).text(text_body).build())
        resp = ms.emails.send(email)
        msg_id = getattr(resp, "message_id", None) or getattr(resp, "id", None) or "mailersend"
        return {"ok": True, "message_id": str(msg_id), "to": recipient}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def mailto_link(subject: str, text_body: str, recipient: str = ALLOWED_ADMIN) -> str:
    """A mailto: URL that opens the user's own mail app with the summary pre-filled.
    Needs no keys/setup — the user just presses Send. Body is trimmed to keep the URL sane."""
    import urllib.parse
    q = urllib.parse.urlencode({"subject": subject, "body": text_body[:1500]})
    return f"mailto:{recipient}?{q}"


def send_summary(subject: str, html_body: str, text_body: str) -> dict:
    """Send one email to the allowed admin inbox. Returns {ok, message_id|error, to}."""
    cfg = load_secrets()
    admin = cfg.get("ADMIN_EMAIL", "")
    if admin != ALLOWED_ADMIN:
        return {"ok": False, "error": f"ADMIN_EMAIL ({admin or 'пусто'}) не совпадает с разрешённым "
                                      f"адресом — отправка отменена (письма только на свой ящик)."}
    # Prefer Gmail SMTP when a Gmail address is set (returns a clear error if the App Password
    # is missing); fall back to MailerSend only if no Gmail is configured at all.
    if cfg.get("GMAIL_ADDRESS"):
        return _send_via_gmail(cfg, subject, html_body, text_body, admin)
    return _send_via_mailersend(cfg, subject, html_body, text_body, admin)


if __name__ == "__main__":
    # One authorized test send to the student's own inbox.
    r = send_summary(
        subject="[Тест] Yessenov Data Lab — ассистент работает",
        html_body="<h2>Проверка связи</h2><p>Это тестовое письмо из приложения-ассистента. "
                  "Если вы его получили — отправка настроена верно.</p>",
        text_body="Проверка связи. Тестовое письмо из приложения-ассистента.",
    )
    print(r)
