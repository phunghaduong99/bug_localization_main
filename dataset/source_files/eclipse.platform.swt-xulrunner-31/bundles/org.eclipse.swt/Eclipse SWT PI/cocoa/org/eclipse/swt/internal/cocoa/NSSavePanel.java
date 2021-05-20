/*******************************************************************************
 * Copyright (c) 2000, 2014 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *    IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.swt.internal.cocoa;

public class NSSavePanel extends NSPanel {

public NSSavePanel() {
	super();
}

public NSSavePanel(long /*int*/ id) {
	super(id);
}

public NSSavePanel(id id) {
	super(id);
}

public NSString filename() {
	long /*int*/ result = OS.objc_msgSend(this.id, OS.sel_filename);
	return result != 0 ? new NSString(result) : null;
}

public long /*int*/ runModal() {
	return OS.objc_msgSend(this.id, OS.sel_runModal);
}

public long /*int*/ runModalForDirectory(NSString path, NSString name) {
	return OS.objc_msgSend(this.id, OS.sel_runModalForDirectory_file_, path != null ? path.id : 0, name != null ? name.id : 0);
}

public static NSSavePanel savePanel() {
	long /*int*/ result = OS.objc_msgSend(OS.class_NSSavePanel, OS.sel_savePanel);
	return result != 0 ? new NSSavePanel(result) : null;
}

public void setAccessoryView(NSView view) {
	OS.objc_msgSend(this.id, OS.sel_setAccessoryView_, view != null ? view.id : 0);
}

public void setAllowedFileTypes(NSArray types) {
	OS.objc_msgSend(this.id, OS.sel_setAllowedFileTypes_, types != null ? types.id : 0);
}

public void setAllowsOtherFileTypes(boolean flag) {
	OS.objc_msgSend(this.id, OS.sel_setAllowsOtherFileTypes_, flag);
}

public void setCanCreateDirectories(boolean flag) {
	OS.objc_msgSend(this.id, OS.sel_setCanCreateDirectories_, flag);
}

public void setDirectory(NSString path) {
	OS.objc_msgSend(this.id, OS.sel_setDirectory_, path != null ? path.id : 0);
}

public void setMessage(NSString message) {
	OS.objc_msgSend(this.id, OS.sel_setMessage_, message != null ? message.id : 0);
}

public void setTitle(NSString title) {
	OS.objc_msgSend(this.id, OS.sel_setTitle_, title != null ? title.id : 0);
}

public void validateVisibleColumns() {
	OS.objc_msgSend(this.id, OS.sel_validateVisibleColumns);
}

public static double /*float*/ minFrameWidthWithTitle(NSString aTitle, long /*int*/ aStyle) {
	return (double /*float*/)OS.objc_msgSend_fpret(OS.class_NSSavePanel, OS.sel_minFrameWidthWithTitle_styleMask_, aTitle != null ? aTitle.id : 0, aStyle);
}

public static long /*int*/ windowNumberAtPoint(NSPoint point, long /*int*/ windowNumber) {
	return OS.objc_msgSend(OS.class_NSSavePanel, OS.sel_windowNumberAtPoint_belowWindowWithWindowNumber_, point, windowNumber);
}

}
