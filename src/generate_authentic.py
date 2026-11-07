from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import string
from datetime import datetime, timedelta

OUTPUT_DIR = Path("data/authentic")
NUM_DOCUMENTS = 500

WIDTH = 1240
HEIGHT = 1754

# Common Windows fonts
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
]

BOLD_FONT_PATHS = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
]

COMPANIES = [
    "NovaTech Solutions",
    "Vertex Systems",
    "BluePeak Technologies",
    "Apex Digital Labs",
    "Orion Enterprises",
    "CloudMatrix Pvt Ltd",
    "Zenith Innovations",
    "PixelCore Solutions",
    "DataSphere Technologies",
    "NextGen Systems",
]

CUSTOMERS = [
    "Rahul Sharma",
    "Arjun Kumar",
    "Priya Reddy",
    "Aditya Singh",
    "Sneha Patel",
    "Rohan Mehta",
    "Kiran Rao",
    "Ananya Gupta",
    "Vikram Das",
    "Neha Verma",
]

PRODUCTS = [
    "Laptop",
    "Wireless Mouse",
    "Mechanical Keyboard",
    "USB-C Hub",
    "Monitor",
    "Web Camera",
    "SSD 1TB",
    "RAM 16GB",
    "Graphics Tablet",
    "Headphones",
    "Power Adapter",
    "External HDD",
]


def find_font(paths, size):
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def random_invoice_number(index):
    return f"INV-{datetime.now().year}-{index:05d}"


def random_date():
    start = datetime(2024, 1, 1)
    date = start + timedelta(days=random.randint(0, 900))
    return date.strftime("%d/%m/%Y")


def random_items():
    items = []

    for _ in range(random.randint(3, 7)):
        product = random.choice(PRODUCTS)
        quantity = random.randint(1, 5)
        price = random.randint(500, 50000)

        items.append(
            {
                "product": product,
                "quantity": quantity,
                "price": price,
                "total": quantity * price,
            }
        )

    return items


def draw_text_center(draw, text, y, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    x = (WIDTH - text_width) // 2

    draw.text(
        (x, y),
        text,
        fill=(30, 30, 30),
        font=font,
    )


def generate_document(index):
    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (250, 250, 250),
    )

    draw = ImageDraw.Draw(image)

    normal = find_font(FONT_PATHS, 25)
    small = find_font(FONT_PATHS, 21)
    bold = find_font(BOLD_FONT_PATHS, 32)
    title = find_font(BOLD_FONT_PATHS, 48)
    table_header = find_font(BOLD_FONT_PATHS, 23)

    margin = 90

    company = random.choice(COMPANIES)
    customer = random.choice(CUSTOMERS)

    invoice_number = random_invoice_number(index)
    invoice_date = random_date()

    # Document border
    draw.rectangle(
        [40, 40, WIDTH - 40, HEIGHT - 40],
        outline=(80, 80, 80),
        width=3,
    )

    # Header
    draw_text_center(
        draw,
        company,
        90,
        title,
    )

    draw_text_center(
        draw,
        "TAX INVOICE",
        160,
        bold,
    )

    # Company details
    draw.text(
        (margin, 250),
        "123 Technology Park",
        fill=(30, 30, 30),
        font=small,
    )

    draw.text(
        (margin, 285),
        "Bengaluru, Karnataka - 560001",
        fill=(30, 30, 30),
        font=small,
    )

    draw.text(
        (margin, 320),
        "GSTIN: 29ABCDE1234F1Z5",
        fill=(30, 30, 30),
        font=small,
    )

    # Invoice details
    right_x = 760

    draw.text(
        (right_x, 250),
        f"Invoice No: {invoice_number}",
        fill=(30, 30, 30),
        font=small,
    )

    draw.text(
        (right_x, 285),
        f"Date: {invoice_date}",
        fill=(30, 30, 30),
        font=small,
    )

    draw.text(
        (right_x, 320),
        "Payment: Online",
        fill=(30, 30, 30),
        font=small,
    )

    # Customer section
    y = 390

    draw.line(
        (margin, y, WIDTH - margin, y),
        fill=(70, 70, 70),
        width=2,
    )

    draw.text(
        (margin, y + 25),
        "BILL TO",
        fill=(20, 20, 20),
        font=bold,
    )

    draw.text(
        (margin, y + 75),
        customer,
        fill=(30, 30, 30),
        font=normal,
    )

    draw.text(
        (margin, y + 115),
        f"{random.randint(10,99)}, MG Road",
        fill=(30, 30, 30),
        font=small,
    )

    draw.text(
        (margin, y + 150),
        "Bengaluru, Karnataka",
        fill=(30, 30, 30),
        font=small,
    )

    # Table
    table_y = 620
    row_height = 75

    columns = [
        ("Item", 100),
        ("Description", 400),
        ("Qty", 700),
        ("Unit Price", 820),
        ("Amount", 1030),
    ]

    # Header background
    draw.rectangle(
        [
            margin,
            table_y,
            WIDTH - margin,
            table_y + row_height,
        ],
        outline=(50, 50, 50),
        fill=(225, 225, 225),
    )

    for text, x in columns:
        draw.text(
            (x, table_y + 22),
            text,
            fill=(20, 20, 20),
            font=table_header,
        )

    items = random_items()

    subtotal = 0

    for i, item in enumerate(items):
        y1 = table_y + row_height * (i + 1)
        y2 = y1 + row_height

        draw.rectangle(
            [margin, y1, WIDTH - margin, y2],
            outline=(100, 100, 100),
        )

        amount = item["total"]
        subtotal += amount

        draw.text(
            (110, y1 + 22),
            str(i + 1),
            fill=(30, 30, 30),
            font=small,
        )

        draw.text(
            (400, y1 + 22),
            item["product"],
            fill=(30, 30, 30),
            font=small,
        )

        draw.text(
            (720, y1 + 22),
            str(item["quantity"]),
            fill=(30, 30, 30),
            font=small,
        )

        draw.text(
            (830, y1 + 22),
            f"₹{item['price']:,}",
            fill=(30, 30, 30),
            font=small,
        )

        draw.text(
            (1030, y1 + 22),
            f"₹{amount:,}",
            fill=(30, 30, 30),
            font=small,
        )

    # Totals
    totals_y = table_y + row_height * (len(items) + 1) + 50

    gst = int(subtotal * 0.18)
    grand_total = subtotal + gst

    draw.text(
        (780, totals_y),
        f"Subtotal: ₹{subtotal:,}",
        fill=(30, 30, 30),
        font=normal,
    )

    draw.text(
        (780, totals_y + 45),
        f"GST (18%): ₹{gst:,}",
        fill=(30, 30, 30),
        font=normal,
    )

    draw.text(
        (780, totals_y + 100),
        f"Grand Total: ₹{grand_total:,}",
        fill=(10, 10, 10),
        font=bold,
    )

    # Footer
    footer_y = HEIGHT - 250

    draw.line(
        (margin, footer_y, WIDTH - margin, footer_y),
        fill=(80, 80, 80),
        width=2,
    )

    draw.text(
        (margin, footer_y + 35),
        "Thank you for your business.",
        fill=(30, 30, 30),
        font=normal,
    )

    draw.text(
        (margin, footer_y + 80),
        "This is a computer-generated invoice.",
        fill=(80, 80, 80),
        font=small,
    )

    # Random slight paper noise
    for _ in range(1200):
        x = random.randint(45, WIDTH - 45)
        y = random.randint(45, HEIGHT - 45)

        shade = random.randint(235, 255)

        draw.point(
            (x, y),
            fill=(shade, shade, shade),
        )

    filename = OUTPUT_DIR / f"synthetic_invoice_{index:04d}.png"

    image.save(
        filename,
        "PNG",
        optimize=True,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Generating {NUM_DOCUMENTS} synthetic documents...")
    print(f"Output: {OUTPUT_DIR}")

    for i in range(1, NUM_DOCUMENTS + 1):
        generate_document(i)

        if i % 25 == 0:
            print(f"Generated {i}/{NUM_DOCUMENTS}")

    print()
    print("DONE!")
    print(f"Generated {NUM_DOCUMENTS} authentic documents.")


if __name__ == "__main__":
    main()