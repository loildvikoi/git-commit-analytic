from contextlib import AsyncExitStack
from typing import Callable, TypeVar, Any

from fastapi import Request
from fastapi.dependencies.utils import (
    get_dependant,
    solve_dependencies,
    is_gen_callable,
    is_async_gen_callable,
    solve_generator,
    is_coroutine_callable,
)
from fastapi.params import Depends

PCallable = TypeVar("PCallable", bound=Callable)


class FastReflector:
    def __init__(self):
        pass

    async def __aenter__(self):
        self.cm = AsyncExitStack()
        self.dependency_cache = {}
        await self.cm.__aenter__()
        return self

    async def __aexit__(self, *exc_details):
        return await self.cm.__aexit__(*exc_details)

    async def resolve(self, callable: PCallable) -> PCallable:
        request = Request(
            {
                "type": "http",
                "headers": [],
                "query_string": "",
                "fastapi_astack": self.cm,
            }
        )

        if callable.__name__ == "Annotated":
            callable = callable.__metadata__[0]

        if isinstance(callable, Depends):
            use_cache = callable.use_cache
            callable = callable.dependency
        else:
            use_cache = True

        dependant = get_dependant(
            path=f"fast_reflector", call=callable, use_cache=use_cache
        )

        dependency_cache = self.dependency_cache

        result = await solve_dependencies(
            request=request,
            dependant=dependant,
            async_exit_stack=self.cm,
            embed_body_fields=False,
            dependency_cache=dependency_cache,
        )

        dependency_cache.update(result.dependency_cache)

        if result.errors:
            raise ValueError(result.errors)

        if dependant.use_cache and dependant.cache_key in dependency_cache:
            solved = dependency_cache[dependant.cache_key]
        elif is_gen_callable(dependant.call) or is_async_gen_callable(dependant.call):
            solved = await solve_generator(
                call=dependant.call, stack=self.cm, sub_values=result.values
            )
        elif is_coroutine_callable(dependant.call):
            solved = await dependant.call(**result.values)
        else:
            solved = dependant.call(**result.values)
        if dependant.cache_key not in dependency_cache:
            dependency_cache[dependant.cache_key] = solved

        return solved

    async def resolve_dependencies(
        self, callable: PCallable, use_cache: bool = True
    ) -> dict[str, Any]:
        request = Request(
            {
                "type": "http",
                "headers": [],
                "query_string": "",
                "fastapi_astack": self.cm,
            }
        )

        dependant = get_dependant(
            path=f"fast_reflector", call=callable, use_cache=use_cache
        )

        dependency_cache = self.dependency_cache

        result = await solve_dependencies(
            request=request,
            dependant=dependant,
            async_exit_stack=self.cm,
            embed_body_fields=False,
            dependency_cache=dependency_cache,
        )

        dependency_cache.update(result.dependency_cache)

        return result.values
