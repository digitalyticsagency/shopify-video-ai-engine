"""Shopify GraphQL Admin API client for fetching promotable products."""
import logging
import re
from dataclasses import dataclass, field

import httpx

from config import settings

logger = logging.getLogger(__name__)

API_VERSION = "2024-10"

PRODUCTS_QUERY = """
query FetchPromotableProducts($first: Int!) {
  products(first: $first, sortKey: UPDATED_AT, reverse: true) {
    edges {
      node {
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
      }
    }
  }
}
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
        self.store_url = (store_url or settings.SHOPIFY_STORE_URL).strip().rstrip("/")
        self.access_token = access_token or settings.SHOPIFY_ACCESS_TOKEN
        self.endpoint = f"https://{self.store_url}/admin/api/{API_VERSION}/graphql.json"

    async def fetch_top_products(self, count: int = 3) -> list[ShopifyProduct]:
        """Fetch the top `count` most recently updated products for video promotion."""
        headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }
        payload = {"query": PRODUCTS_QUERY, "variables": {"first": count}}

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

        try:
            edges = body["data"]["products"]["edges"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Shopify response shape: %s", body)
            raise ShopifyClientError("Unexpected Shopify response shape") from exc

        products: list[ShopifyProduct] = []
        for edge in edges:
            node = edge.get("node", {})
            products.append(self._parse_product(node))

        logger.info("Fetched %d products from Shopify", len(products))
        return products

    @staticmethod
    def _parse_product(node: dict) -> ShopifyProduct:
        title = node.get("title") or "Untitled Product"
        description = strip_html(node.get("descriptionHtml"))
        online_store_url = node.get("onlineStoreUrl")

        price_info = (node.get("priceRangeV2") or {}).get("minVariantPrice") or {}
        price = price_info.get("amount", "0.00")
        currency_code = price_info.get("currencyCode", "USD")

        image_edges = (node.get("images") or {}).get("edges") or []
        image_urls = [e["node"]["url"] for e in image_edges if e.get("node", {}).get("url")]

        return ShopifyProduct(
            title=title,
            description=description,
            online_store_url=online_store_url,
            price=price,
            currency_code=currency_code,
            image_urls=image_urls,
        )
