"""Shopify GraphQL Admin API client for fetching promotable products."""
import logging
import re
from dataclasses import dataclass, field

import httpx

import runtime_settings

logger = logging.getLogger(__name__)

API_VERSION = "2024-10"

_PRODUCT_FIELDS = """
        id
        title
        descriptionHtml
        onlineStoreUrl
        priceRangeV2 {
          minVariantPrice {
            amount
            currencyCode
          }
        }
        images(first: 3) {
          edges {
            node {
              url
            }
          }
        }
"""

PRODUCTS_QUERY = f"""
query FetchPromotableProducts($first: Int!) {{
  products(first: $first, sortKey: UPDATED_AT, reverse: true) {{
    edges {{
      node {{
{_PRODUCT_FIELDS}
      }}
    }}
  }}
}}
"""

PRODUCTS_BY_ID_QUERY = f"""
query FetchProductsByIds($ids: [ID!]!) {{
  nodes(ids: $ids) {{
    ... on Product {{
{_PRODUCT_FIELDS}
    }}
  }}
}}
"""

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(raw_html: str | None) -> str:
    """Strip HTML tags from a product description and collapse whitespace."""
    if not raw_html:
        return ""
    text = _TAG_RE.sub(" ", raw_html)
    return _WHITESPACE_RE.sub(" ", text).strip()


@dataclass
class ShopifyProduct:
    id: str
    title: str
    description: str
    online_store_url: str | None
    price: str
    currency_code: str
    image_urls: list[str] = field(default_factory=list)


class ShopifyClientError(Exception):
    """Raised when the Shopify Admin API request fails or returns errors."""


class ShopifyClient:
    """Class-based client for querying the Shopify GraphQL Admin API."""

    def __init__(self, store_url: str | None = None, access_token: str | None = None) -> None:
        resolved_store_url = store_url or runtime_settings.get_shopify_store_url()
        resolved_access_token = access_token or runtime_settings.get_shopify_access_token()
        if not resolved_store_url or not resolved_access_token:
            raise ShopifyClientError(
                "Shopify store not connected. Set the store URL and access token at /settings "
                "or via SHOPIFY_STORE_URL / SHOPIFY_ACCESS_TOKEN in .env."
            )
        self.store_url = resolved_store_url.strip().rstrip("/")
        self.access_token = resolved_access_token
        self.endpoint = f"https://{self.store_url}/admin/api/{API_VERSION}/graphql.json"

    async def fetch_products(self, count: int = 20) -> list[ShopifyProduct]:
        """Fetch the `count` most recently updated products, for browsing or auto-selection."""
        body = await self._graphql(PRODUCTS_QUERY, {"first": count})

        try:
            edges = body["data"]["products"]["edges"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Shopify response shape: %s", body)
            raise ShopifyClientError("Unexpected Shopify response shape") from exc

        products = [self._parse_product(edge.get("node", {})) for edge in edges]
        logger.info("Fetched %d products from Shopify", len(products))
        return products

    async def fetch_top_products(self, count: int = 3) -> list[ShopifyProduct]:
        """Fetch the top `count` most recently updated products for video promotion."""
        return await self.fetch_products(count)

    async def fetch_products_by_ids(self, product_ids: list[str]) -> list[ShopifyProduct]:
        """Fetch specific products by their Shopify GID, in the order given."""
        if not product_ids:
            return []

        body = await self._graphql(PRODUCTS_BY_ID_QUERY, {"ids": product_ids})

        try:
            nodes = body["data"]["nodes"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Shopify response shape: %s", body)
            raise ShopifyClientError("Unexpected Shopify response shape") from exc

        products = [self._parse_product(node) for node in nodes if node]
        logger.info("Fetched %d products by id from Shopify", len(products))
        return products

    async def _graphql(self, query: str, variables: dict) -> dict:
        headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }
        payload = {"query": query, "variables": variables}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.endpoint, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Shopify API returned HTTP %s: %s", exc.response.status_code, exc.response.text)
            raise ShopifyClientError(f"Shopify API HTTP error: {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error("Shopify API request failed: %s", exc)
            raise ShopifyClientError(f"Shopify API request failed: {exc}") from exc

        body = response.json()
        if "errors" in body:
            logger.error("Shopify GraphQL errors: %s", body["errors"])
            raise ShopifyClientError(f"Shopify GraphQL errors: {body['errors']}")
        return body

    @staticmethod
    def _parse_product(node: dict) -> ShopifyProduct:
        product_id = node.get("id") or ""
        title = node.get("title") or "Untitled Product"
        description = strip_html(node.get("descriptionHtml"))
        online_store_url = node.get("onlineStoreUrl")

        price_info = (node.get("priceRangeV2") or {}).get("minVariantPrice") or {}
        price = price_info.get("amount", "0.00")
        currency_code = price_info.get("currencyCode", "USD")

        image_edges = (node.get("images") or {}).get("edges") or []
        image_urls = [e["node"]["url"] for e in image_edges if e.get("node", {}).get("url")]

        return ShopifyProduct(
            id=product_id,
            title=title,
            description=description,
            online_store_url=online_store_url,
            price=price,
            currency_code=currency_code,
            image_urls=image_urls,
        )
