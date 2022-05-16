import fitz

doc = fitz.open(r'Development\imageTest\Scan.pdf')
for pg in range(doc.pageCount):
    page = doc[pg]
    rotate = int(0)
    zoom_x = 2.0
    zoom_y = 2.0
    trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
    pm = page.get_pixmap(matrix= trans, alpha = False)
    pm.save('%s.png' % pg)