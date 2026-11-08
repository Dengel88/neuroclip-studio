"""«Создай видео»: the paid service on top of the storyboard pipeline.

The pipeline (``pipeline.py``) turns a brief into a storyboard. This package
turns a storyboard into an *order*: who asked for it, what it costs, whether
it is paid or the free one for subscribing, and where it is in production.
Video generation and assembly stay manual for now; the operator moves an
order through its statuses and the client follows along in the cabinet and
in Telegram.
"""
