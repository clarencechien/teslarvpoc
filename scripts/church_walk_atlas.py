"""Refresh registered-photo tiles while retaining every GLB accessor and material assignment."""

import json
import struct
import zipfile
from pathlib import Path
import cv2
import numpy as np

from build_christ_church import read_photo
from church_rail_repair import camera_list


def thumbnail(image):
    return cv2.resize(image, (24, 18), interpolation=cv2.INTER_AREA).astype("f4")


def repair_atlases(data, source, cache, output):
    cameras = camera_list(data)
    ids = []
    thumbs = []
    with zipfile.ZipFile(data / "images.zip") as archive:
        for c, cam in enumerate(cameras):
            for i in range(0, len(cam), 4):
                if 1710755131 < float(Path(cam[i]["file_path"]).stem) < 1710755444:
                    ids.append(c * 1000 + i)
                    thumbs.append(thumbnail(read_photo(archive, cam[i])))
    thumbs = np.asarray(thumbs)
    for name in ("christ-church.glb", "christ-church-mobile.glb"):
        raw = (source / name).read_bytes()
        size = struct.unpack_from("<I", raw, 12)[0]
        doc = json.loads(raw[20 : 20 + size])
        binary = raw[28 + size :]
        previous = doc["asset"].get("extras", {}).get("walkPhotoCalibration", {}).get("tileSources", {})
        replacements = {}
        count = 0
        sources = {}

        for mat in doc["materials"]:
            if mat.get("name") != "Registered photographs":
                continue
            index = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            buf = doc["images"][doc["textures"][index]["source"]]["bufferView"]
            item = doc["bufferViews"][buf]
            off = item["byteOffset"]
            pixels = cv2.imdecode(np.frombuffer(binary[off : off + item["byteLength"]], "u1"), 1)
            h, w = np.array(pixels.shape[:2]) // 4
            tile_ids = []
            for y in range(0, pixels.shape[0], h):
                for x in range(0, pixels.shape[1], w):
                    tile = pixels[y + 2 : y + h - 2, x + 2 : x + w - 2]
                    if str(buf) in previous:
                        ident = previous[str(buf)][len(tile_ids)]
                    elif tile.mean() < 1 and tile.max() < 24:
                        ident = None
                    else:
                        error = abs(thumbs - thumbnail(tile)).mean((1, 2, 3))
                        slot = error.argmin()
                        if error[slot] > 3:
                            raise ValueError(
                                f"Unmatched photographic tile in {name}, error {error[slot]:.2f}"
                            )
                        ident = ids[slot]
                    tile_ids.append(ident)
                    if ident is None:
                        continue
                    photo = cv2.imread(str(cache / f"clean-{ident}.webp"))
                    if photo is None:
                        raise ValueError(f"Missing corrected photograph {ident}")
                    tile = cv2.resize(photo, (w - 4, h - 4), interpolation=cv2.INTER_AREA)
                    pixels[y : y + h, x : x + w] = cv2.copyMakeBorder(
                        tile, 2, 2, 2, 2, cv2.BORDER_REPLICATE
                    )
                    count += 1
            sources[str(buf)] = tile_ids
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 91])
            if not ok:
                raise ValueError("Failed to encode corrected atlas")
            replacements[buf] = encoded.tobytes()
        blob = bytearray()
        for i, item in enumerate(doc["bufferViews"]):
            off = item.get("byteOffset", 0)
            value = replacements.get(i, binary[off : off + item["byteLength"]])
            blob.extend(bytes(-len(blob) % 4))
            item.update(byteOffset=len(blob), byteLength=len(value))
            blob.extend(value)
        blob.extend(bytes(-len(blob) % 4))
        doc["buffers"][0]["byteLength"] = len(blob)
        doc["asset"].setdefault("extras", {})["walkPhotoCalibration"] = dict(
            registeredTiles=count, tileSources=sources
        )
        meta = json.dumps(doc, separators=(",", ":")).encode()
        meta += b" " * (-len(meta) % 4)
        header = struct.pack("<4sII", b"glTF", 2, 28 + len(meta) + len(blob))
        (output / name).write_bytes(
            header
            + struct.pack("<I4s", len(meta), b"JSON")
            + meta
            + struct.pack("<I4s", len(blob), b"BIN\0")
            + blob
        )
        print(name, "registered tiles", count, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "source", "cache", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Choose a separate output directory for the rebuilt GLBs")
    args.output.mkdir(parents=True, exist_ok=True)
    repair_atlases(**vars(args))
