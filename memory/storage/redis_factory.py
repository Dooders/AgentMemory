"""Redis client factory for agent memory system.

This module provides a factory to create Redis clients with the option
to use either real Redis or MockRedis for testing or local development.
"""

import logging
from typing import Optional, Type, Union

import redis

from agent_memory.storage.redis_client import ResilientRedisClient
from agent_memory.storage.async_redis_client import AsyncResilientRedisClient
from agent_memory.storage.mockredis import MockRedis

logger = logging.getLogger(__name__)


class RedisFactory:
    """Factory for creating Redis clients.
    
    This class provides methods to create either real Redis clients
    or MockRedis clients for testing and local development.
    """
    
    @staticmethod
    def create_client(
        client_name: str,
        use_mock: bool = False,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        circuit_threshold: int = 3,
        circuit_reset_timeout: int = 300,
        critical_retry_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 30.0,
        max_connections: int = 10,
        health_check_interval: int = 30,
        use_resilient_client: bool = True,
    ) -> ResilientRedisClient:
        """Create a Redis client.
        
        Args:
            client_name: Name of this Redis client instance
            use_mock: Whether to use MockRedis instead of real Redis
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            socket_timeout: Timeout for Redis operations
            socket_connect_timeout: Timeout for connection attempts
            circuit_threshold: Failures before circuit breaker opens
            circuit_reset_timeout: Seconds before circuit breaker resets
            critical_retry_attempts: Number of immediate retries for critical operations
            retry_base_delay: Base delay for exponential backoff (seconds)
            retry_max_delay: Maximum delay for exponential backoff (seconds)
            max_connections: Maximum number of connections in the pool
            health_check_interval: Seconds between health checks of idle connections
            use_resilient_client: Whether to use ResilientRedisClient directly
            
        Returns:
            Redis client instance
        """
        # Return ResilientRedisClient instance directly when use_resilient_client=True
        if use_resilient_client:
            logger.info(f"Creating ResilientRedisClient '{client_name}' directly (host={host}, port={port}, db={db})")
            return ResilientRedisClient(
                client_name=client_name,
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                circuit_threshold=circuit_threshold,
                circuit_reset_timeout=circuit_reset_timeout,
                critical_retry_attempts=critical_retry_attempts,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
                max_connections=max_connections,
                health_check_interval=health_check_interval,
            )
        
        if use_mock:
            logger.info(f"Creating MockRedis client '{client_name}'")
            # Override Redis class to use MockRedis
            original_redis = redis.Redis
            redis.Redis = MockRedis
            
            try:
                # Create client with MockRedis
                client = ResilientRedisClient(
                    client_name=client_name,
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    circuit_threshold=circuit_threshold,
                    circuit_reset_timeout=circuit_reset_timeout,
                    critical_retry_attempts=critical_retry_attempts,
                    retry_base_delay=retry_base_delay,
                    retry_max_delay=retry_max_delay,
                    max_connections=max_connections,
                    health_check_interval=health_check_interval,
                )
                return client
            finally:
                # Restore original Redis class
                redis.Redis = original_redis
        else:
            logger.info(f"Creating real Redis client '{client_name}' (host={host}, port={port}, db={db})")
            # Create client with real Redis
            return ResilientRedisClient(
                client_name=client_name,
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                circuit_threshold=circuit_threshold,
                circuit_reset_timeout=circuit_reset_timeout,
                critical_retry_attempts=critical_retry_attempts,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
                max_connections=max_connections,
                health_check_interval=health_check_interval,
            )
    
    @staticmethod
    async def create_async_client(
        client_name: str,
        use_mock: bool = False,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        circuit_threshold: int = 3,
        circuit_reset_timeout: int = 300,
        critical_retry_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 30.0,
        max_connections: int = 10,
        health_check_interval: int = 30,
    ) -> AsyncResilientRedisClient:
        """Create an async Redis client.
        
        Args:
            client_name: Name of this Redis client instance
            use_mock: Whether to use MockRedis instead of real Redis
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            socket_timeout: Timeout for Redis operations
            socket_connect_timeout: Timeout for connection attempts
            circuit_threshold: Failures before circuit breaker opens
            circuit_reset_timeout: Seconds before circuit breaker resets
            critical_retry_attempts: Number of immediate retries for critical operations
            retry_base_delay: Base delay for exponential backoff (seconds)
            retry_max_delay: Maximum delay for exponential backoff (seconds)
            max_connections: Maximum number of connections in the pool
            health_check_interval: Seconds between health checks of idle connections
            
        Returns:
            Async Redis client instance
        """
        if use_mock:
            logger.info(f"Creating MockRedis client '{client_name}'")
            
            # Create a custom ResilientRedisClient that uses MockRedis
            class MockResilientRedisClient(ResilientRedisClient):
                def _create_redis_client(self):
                    # Override the method to return a MockRedis instance
                    return MockRedis(
                        host=self.connection_params["connection_pool"].connection_kwargs.get("host"),
                        port=self.connection_params["connection_pool"].connection_kwargs.get("port"),
                        db=self.connection_params["connection_pool"].connection_kwargs.get("db"),
                        password=self.connection_params["connection_pool"].connection_kwargs.get("password"),
                    )
            
            # Create client with MockRedis via dependency injection
            return MockResilientRedisClient(
                client_name=client_name,
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                circuit_threshold=circuit_threshold,
                circuit_reset_timeout=circuit_reset_timeout,
                critical_retry_attempts=critical_retry_attempts,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
                max_connections=max_connections,
                health_check_interval=health_check_interval,
            )
        else:
            logger.info(f"Creating real async Redis client '{client_name}' (host={host}, port={port}, db={db})")
            # Create client with real Redis
            client = AsyncResilientRedisClient(
                client_name=client_name,
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                circuit_threshold=circuit_threshold,
                circuit_reset_timeout=circuit_reset_timeout,
                critical_retry_attempts=critical_retry_attempts,
                retry_base_delay=retry_base_delay,
                retry_max_delay=retry_max_delay,
                max_connections=max_connections,
                health_check_interval=health_check_interval,
            )
            return client 