## Longaxis Endpoint

# Returns a point coordinates for the crown and the root of every tooth if both appear in the periapical image.

longaxis_response = requests.post(
    BASE_URL + "/periapicals/longaxis",
    json={
        "base64_image": encode_image(periapical_image),
    },
    headers=headers,
)

pdata = longaxis_response.json()
pdata["entities"] = pdata["entities"][:3]
pp.pprint(pdata)


from vis import draw_longaxis_output

dimage = draw_longaxis_output(
    periapical_image, longaxis_response.json()["entities"], draw_axis=True, th=0.001
)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(dimage)
ax.axis("off")

tele_image = Image.open("images/tele0.jpg").convert("L")


cefbot_response = requests.post(
    BASE_URL + "/cephalometry/cefbot",
    json={
        "base64_image": encode_image(tele_image),
    },
    headers=headers,
)

cefbot_response.json()
