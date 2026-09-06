"""The Stack object: one namespace composed from focused mixins.
Boto3 clients are reached as attributes (stack.s3, stack.lam, stack.ecs, ...),
resolved lazily by Base.__getattr__.
"""
from .apis import ApiMixin
from .base import Base
from .database import DatabaseMixin
from .frontend import FrontendMixin
from .lambdas import LambdaMixin
from .messaging import MessagingMixin
from .storage import StorageMixin
from .webapi import WebApiMixin
class Stack(FrontendMixin, WebApiMixin, ApiMixin, LambdaMixin, MessagingMixin,
            DatabaseMixin, StorageMixin, Base):
    pass