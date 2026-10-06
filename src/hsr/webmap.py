"""Interactive planner map (Folium / Leaflet)."""
from __future__ import annotations

import folium
import numpy as np
from branca.colormap import LinearColormap

RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]  # sequential blue


def make_map(sites, prob, realized, path, title="Housing site realization"):
    cent = sites.to_crs(2230).geometry.centroid.to_crs(4326)
    vmax = float(np.quantile(prob, 0.98))
    cmap = LinearColormap(RAMP, vmin=0, vmax=vmax, caption="Predicted probability of housing approval")
    m = folium.Map(location=[32.78, -117.13], zoom_start=11, tiles="OpenStreetMap",
                   control_scale=True)
    pred = folium.FeatureGroup(name="Predicted probability", show=True)
    built = folium.FeatureGroup(name="Actually approved (outcome)", show=False)
    order = np.argsort(prob.values)                       # draw high-probability last
    for i in order:
        pt, p = cent.iloc[i], float(prob.iloc[i])
        cap = int(max(sites.NetUnits.iloc[i], 0))
        popup = (f"<b>{sites.CPNAME.iloc[i]}</b><br>APN {sites.APN_8.iloc[i]}<br>"
                 f"Existing use: {sites.Extg_Landuse.iloc[i]}<br>"
                 f"Claimed net units: {cap}<br>Predicted probability: <b>{p:.0%}</b><br>"
                 f"Expected units: {p * cap:.1f}<br>"
                 f"Approved after adoption: {'Yes' if realized.iloc[i] else 'No'}")
        folium.CircleMarker([pt.y, pt.x], radius=float(np.clip(2 + np.log1p(cap), 2, 9)),
                            color=cmap(min(p, vmax)), weight=0.5, fill=True,
                            fill_color=cmap(min(p, vmax)), fill_opacity=0.85,
                            popup=folium.Popup(popup, max_width=260)).add_to(pred)
        if realized.iloc[i]:
            folium.CircleMarker([pt.y, pt.x], radius=4, color="#eb6834", weight=1,
                                fill=True, fill_opacity=0.9).add_to(built)
    pred.add_to(m)
    built.add_to(m)
    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;'
        f'padding:6px 10px;border-radius:6px;font:600 14px sans-serif;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.2)">{title}</div>'))
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
