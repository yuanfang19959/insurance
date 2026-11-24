import jwt
from app.core.config import settings


def test_jwt():
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJobS1pbnN1cmFuY2UiLCJzdWIiOiIyIiwidXNlcl9pZCI6MiwidXNlcm5hbWUiOiIxNTI2NzA4MTAxMiIsInR5cCI6ImFjY2VzcyIsImlhdCI6MTc5NDY0MzM5NywiZXhwIjoxNzk0NjQ0Mjk3fQ.fUiTk_suU7B_NlGwduyA53141vOhAz85xRVP8RxvceQ"

    payload = jwt.decode(token, settings.jwt_secret, algorithms=['HS256'])

    print(payload)